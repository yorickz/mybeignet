/*
 * Copyright © 2012 Intel Corporation
 *
 * This library is free software; you can redistribute it and/or
 * modify it under the terms of the GNU Lesser General Public
 * License as published by the Free Software Foundation; either
 * version 2 of the License, or (at your option) any later version.
 *
 * This library is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
 * Lesser General Public License for more details.
 *
 * You should have received a copy of the GNU Lesser General Public
 * License along with this library. If not, see <http://www.gnu.org/licenses/>.
 *
 * Author: Benjamin Segovia <benjamin.segovia@intel.com>
 */

/**
 * \file llvm_to_gen.cpp
 * \author Benjamin Segovia <benjamin.segovia@intel.com>
 */

#include "llvm/Config/config.h"
#if LLVM_VERSION_MAJOR == 3 && LLVM_VERSION_MINOR <= 2
#include "llvm/LLVMContext.h"
#include "llvm/Module.h"
#include "llvm/DataLayout.h"
#else
#include "llvm/IR/LLVMContext.h"
#include "llvm/IR/Module.h"
#include "llvm/IR/DataLayout.h"
#endif  /* LLVM_VERSION_MINOR <= 2 */
#include "llvm/PassManager.h"
#include "llvm/Pass.h"
#include "llvm/Analysis/Passes.h"
#include "llvm/Transforms/IPO.h"
#include "llvm/Target/TargetLibraryInfo.h"
#include "llvm/ADT/Triple.h"
#if LLVM_VERSION_MAJOR == 3 && LLVM_VERSION_MINOR <= 2
#include "llvm/Support/IRReader.h"
#else
#include "llvm/IRReader/IRReader.h"
#include "llvm/Support/SourceMgr.h"
#endif  /* LLVM_VERSION_MINOR <= 2 */
#include "llvm/Support/raw_ostream.h"
#include "llvm/Transforms/Scalar.h"

#if LLVM_VERSION_MAJOR == 3 && LLVM_VERSION_MINOR >=5
#include "llvm/IR/IRPrintingPasses.h"
#include "llvm/IR/Verifier.h"
#else
#include "llvm/Analysis/Verifier.h"
#include "llvm/Assembly/PrintModulePass.h"
#endif

#include "llvm/Analysis/CFGPrinter.h"
#include "llvm/llvm_gen_backend.hpp"
#include "llvm/llvm_to_gen.hpp"
#include "sys/cvar.hpp"
#include "sys/platform.hpp"
#include "ir/unit.hpp"
#include "ir/structural_analysis.hpp"

#include <sys/types.h>
#include <sys/stat.h>
#include <fcntl.h>
#include <memory>

namespace gbe
{
  BVAR(OCL_OUTPUT_LLVM, false);
  BVAR(OCL_OUTPUT_CFG, false);
  BVAR(OCL_OUTPUT_CFG_ONLY, false);
  BVAR(OCL_OUTPUT_LLVM_BEFORE_EXTRA_PASS, false);
  using namespace llvm;

  void runFuntionPass(Module &mod, TargetLibraryInfo *libraryInfo)
  {
    FunctionPassManager FPM(&mod);
    FPM.add(new DataLayout(&mod));
#if LLVM_VERSION_MAJOR == 3 && LLVM_VERSION_MINOR >=5
    FPM.add(createVerifierPass(true));
#else
    FPM.add(createVerifierPass());
#endif
    FPM.add(new TargetLibraryInfo(*libraryInfo));
    FPM.add(createTypeBasedAliasAnalysisPass());
    FPM.add(createBasicAliasAnalysisPass());
    FPM.add(createCFGSimplificationPass());
    FPM.add(createSROAPass());
    FPM.add(createEarlyCSEPass());
    FPM.add(createLowerExpectIntrinsicPass());

    FPM.doInitialization();
    for (Module::iterator I = mod.begin(),
           E = mod.end(); I != E; ++I)
      if (!I->isDeclaration())
        FPM.run(*I);
    FPM.doFinalization();
  }

  void runModulePass(Module &mod, TargetLibraryInfo *libraryInfo, int optLevel)
  {
    llvm::PassManager MPM;

    MPM.add(new DataLayout(&mod));
    MPM.add(new TargetLibraryInfo(*libraryInfo));
    MPM.add(createTypeBasedAliasAnalysisPass());
    MPM.add(createBasicAliasAnalysisPass());
    MPM.add(createGlobalOptimizerPass());     // Optimize out global vars

    MPM.add(createIPSCCPPass());              // IP SCCP
    MPM.add(createDeadArgEliminationPass());  // Dead argument elimination

    MPM.add(createInstructionCombiningPass());// Clean up after IPCP & DAE
    MPM.add(createCFGSimplificationPass());   // Clean up after IPCP & DAE
    MPM.add(createPruneEHPass());             // Remove dead EH info
    MPM.add(createBarrierNodupPass(false));   // remove noduplicate fnAttr before inlining.
    MPM.add(createFunctionInliningPass(200000));
    MPM.add(createBarrierNodupPass(true));    // restore noduplicate fnAttr after inlining.
    MPM.add(createFunctionAttrsPass());       // Set readonly/readnone attrs

    //MPM.add(createScalarReplAggregatesPass(64, true, -1, -1, 64))
    if(optLevel > 0)
      MPM.add(createSROAPass(/*RequiresDomTree*/ false));
    MPM.add(createEarlyCSEPass());              // Catch trivial redundancies
    MPM.add(createJumpThreadingPass());         // Thread jumps.
    MPM.add(createCorrelatedValuePropagationPass()); // Propagate conditionals
    MPM.add(createCFGSimplificationPass());     // Merge & remove BBs
    MPM.add(createInstructionCombiningPass());  // Combine silly seq's

    MPM.add(createTailCallEliminationPass());   // Eliminate tail calls
    MPM.add(createCFGSimplificationPass());     // Merge & remove BBs
    MPM.add(createReassociatePass());           // Reassociate expressions
    MPM.add(createLoopRotatePass());            // Rotate Loop
    MPM.add(createLICMPass());                  // Hoist loop invariants
    MPM.add(createLoopUnswitchPass(true));
    MPM.add(createInstructionCombiningPass());
    MPM.add(createIndVarSimplifyPass());        // Canonicalize indvars
    MPM.add(createLoopIdiomPass());             // Recognize idioms like memset.
    MPM.add(createLoopDeletionPass());          // Delete dead loops
    MPM.add(createLoopUnrollPass());          // Unroll small loops
    if(optLevel > 0)
      MPM.add(createGVNPass());                 // Remove redundancies
    MPM.add(createMemCpyOptPass());             // Remove memcpy / form memset
    MPM.add(createSCCPPass());                  // Constant prop with SCCP

    // Run instcombine after redundancy elimination to exploit opportunities
    // opened up by them.
    MPM.add(createInstructionCombiningPass());
    MPM.add(createJumpThreadingPass());         // Thread jumps
    MPM.add(createCorrelatedValuePropagationPass());
    MPM.add(createDeadStoreEliminationPass());  // Delete dead stores
    MPM.add(createAggressiveDCEPass());         // Delete dead instructions
    MPM.add(createCFGSimplificationPass()); // Merge & remove BBs
    MPM.add(createInstructionCombiningPass());  // Clean up after everything.
    MPM.add(createStripDeadPrototypesPass()); // Get rid of dead prototypes
    if(optLevel > 0) {
      MPM.add(createGlobalDCEPass());         // Remove dead fns and globals.
      MPM.add(createConstantMergePass());     // Merge dup global constants
    }

    MPM.run(mod);
  }

  bool llvmToGen(ir::Unit &unit, const char *fileName, int optLevel)
  {
    // Get the global LLVM context
    llvm::LLVMContext& c = llvm::getGlobalContext();
    std::string errInfo;
    std::unique_ptr<llvm::raw_fd_ostream> o = NULL;
    if (OCL_OUTPUT_LLVM_BEFORE_EXTRA_PASS || OCL_OUTPUT_LLVM)
      o = std::unique_ptr<llvm::raw_fd_ostream>(new llvm::raw_fd_ostream(fileno(stdout), false));

    // Get the module from its file
    llvm::SMDiagnostic Err;
    std::auto_ptr<Module> M;
    M.reset(ParseIRFile(fileName, Err, c));
    if (M.get() == 0) return false;
    Module &mod = *M.get();

    Triple TargetTriple(mod.getTargetTriple());
    TargetLibraryInfo *libraryInfo = new TargetLibraryInfo(TargetTriple);
    libraryInfo->disableAllFunctions();

    runFuntionPass(mod, libraryInfo);
    runModulePass(mod, libraryInfo, optLevel);

    llvm::PassManager passes;
    passes.add(new DataLayout(&mod));
    // Print the code before further optimizations
    if (OCL_OUTPUT_LLVM_BEFORE_EXTRA_PASS)
#if LLVM_VERSION_MAJOR == 3 && LLVM_VERSION_MINOR >= 5
      passes.add(createPrintModulePass(*o));
#else
      passes.add(createPrintModulePass(&*o));
#endif
    passes.add(createIntrinsicLoweringPass());
    passes.add(createFunctionInliningPass(200000));
    passes.add(createScalarReplAggregatesPass()); // Break up allocas
    passes.add(createLoadStoreOptimizationPass());
    passes.add(createRemoveGEPPass(unit));
    passes.add(createConstantPropagationPass());
    passes.add(createLowerSwitchPass());
    passes.add(createPromoteMemoryToRegisterPass());
    passes.add(createGVNPass());                  // Remove redundancies
    passes.add(createScalarizePass());        // Expand all vector ops
    passes.add(createDeadInstEliminationPass());  // Remove simplified instructions
    passes.add(createCFGSimplificationPass());     // Merge & remove BBs
    passes.add(createScalarizePass());        // Expand all vector ops

    if(OCL_OUTPUT_CFG)
      passes.add(createCFGPrinterPass());
    if(OCL_OUTPUT_CFG_ONLY)
      passes.add(createCFGOnlyPrinterPass());
    passes.add(createGenPass(unit));

    // Print the code extra optimization passes
    if (OCL_OUTPUT_LLVM)
#if LLVM_VERSION_MAJOR == 3 && LLVM_VERSION_MINOR >= 5
      passes.add(createPrintModulePass(*o));
#else
      passes.add(createPrintModulePass(&*o));
#endif
    passes.run(mod);

#ifdef TRANSFORM_UNSTRUCTURE
    const ir::Unit::FunctionSet& fs = unit.getFunctionSet();
    ir::Unit::FunctionSet::const_iterator iter = fs.begin();
    while(iter != fs.end())
    {
      analysis::ControlTree *ct = new analysis::ControlTree(iter->second);
      ct->analyze();
      delete ct;
      iter++;
    }
#endif

    return true;
  }
} /* namespace gbe */
