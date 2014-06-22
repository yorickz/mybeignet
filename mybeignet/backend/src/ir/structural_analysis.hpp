#ifndef __STRUCTURAL_ANALYSIS_HPP__
#define __STRUCTURAL_ANALYSIS_HPP__

#include "ir/unit.hpp"
#include "ir/function.hpp"
#include "ir/instruction.hpp"

#include <iostream>
#include <unordered_set>
#include <unordered_map>
#include <vector>
#include <map>
#include <list>
#include <algorithm>
#include <set>
#define TRANSFORM_UNSTRUCTURE

namespace analysis
{
  using namespace std;
  using namespace gbe;

  enum RegionType
  {
    BasicBlock = 0,
    Block,
    IfThen,
    IfElse,
    SelfLoop,
    WhileLoop,
    NaturalLoop
  } ;

  /* control tree virtual node */
  class Node;

  typedef unordered_set<Node *> NodeSet;
  typedef list<Node *> NodeList;
  typedef std::vector<Node *> NodeVector;

  /* control tree virtual node */
  class Node
  {
  public:
    Node(RegionType rtype, const NodeList& children): has_barrier(false), mark(false), canBeHandled(true)
    {
      this->rtype = rtype;
      this->children = children;
    }
    virtual ~Node() {}
    NodeSet& preds() { return pred; }
    NodeSet& succs() { return succ; }
    Node*& fallthrough() { return fall_through; }
    bool& hasBarrier() { return has_barrier; }
    RegionType type() { return rtype; }
    virtual ir::BasicBlock* getEntry()
    {
      return (*(children.begin()))->getEntry();
    }
    virtual ir::BasicBlock* getExit()
    {
      return (*(children.rbegin()))->getExit();
    }

  public:
    RegionType rtype;
    NodeSet pred;
    NodeSet succ;
    NodeList children;
    Node* fall_through;
    bool has_barrier;
    bool mark;
    bool canBeHandled;
    //label is for debug
    int label;
  };

  /* represents basic block */
  class BasicBlockNode : public Node
  {
  public:
    BasicBlockNode(ir::BasicBlock *p_bb) : Node(BasicBlock, NodeList()) { this->p_bb = p_bb; }
    virtual ~BasicBlockNode() {}
    ir::BasicBlock* getBasicBlock() { return p_bb; }
    virtual ir::BasicBlock* getEntry() { return p_bb; }
    virtual ir::BasicBlock* getExit() { return p_bb; }
    virtual ir::BasicBlock* getFirstBB() { return p_bb; }
  private:
    ir::BasicBlock *p_bb;
  };

  /* a sequence of nodes */
  class BlockNode : public Node
  {
  public:
    BlockNode(NodeList& children) : Node(Block, children) {}
    virtual ~BlockNode(){}
  };

  /* If-Then structure node */
  class IfThenNode : public Node
  {
  public:
    IfThenNode(Node* cond, Node* ifTrue) : Node(IfThen, BuildChildren(cond, ifTrue)) {}
    virtual ~IfThenNode() {}

  private:
    const NodeList BuildChildren(Node* cond, Node* ifTrue)
    {
      NodeList children;
      children.push_back(cond);
      children.push_back(ifTrue);
      return children;
    }
  };

  /* If-Else structure node */
  class IfElseNode : public Node
  {
  public:
    IfElseNode(Node* cond, Node* ifTrue, Node* ifFalse) : Node(IfElse, BuildChildren(cond, ifTrue, ifFalse)) {}
    virtual ~IfElseNode() {}

  private:
    const NodeList BuildChildren(Node* cond, Node* ifTrue, Node* ifFalse)
    {
      NodeList children;
      children.push_back(cond);
      children.push_back(ifTrue);
      children.push_back(ifFalse);
      return children;
    }
  };
#if 0
  /* Self loop structure node */
  class SelfLoopNode : public Node
  {
  public:
    SelfLoopNode(Node* node) : Node(SelfLoop, BuildChildren(node)) {}
    virtual ~SelfLoopNode() {}
    virtual ir::BasicBlock* getEntry()
    {
      return (*(children.begin()))->getEntry();
    }
    virtual ir::BasicBlock* getExit()
    {
      return (*(children.begin()))->getExit();
    }

  private:
    const NodeList BuildChildren(Node *node)
    {
      NodeList children;
      children.push_back(node);
      return children;
    }
  };

  /* While loop structure node */
  class WhileLoopNode : public Node
  {
  public:
    WhileLoopNode(Node* cond, Node* execute) : Node(WhileLoop, BuildChildren(cond, execute)) {}
    virtual ~WhileLoopNode() {}
    virtual ir::BasicBlock* getEntry()
    {
      return (*(children.begin()))->getEntry();
    }
    virtual ir::BasicBlock* getExit()
    {
      return (*(children.begin()))->getExit();
    }

  private:
    const NodeList BuildChildren(Node* cond, Node* execute)
    {
      NodeList children;
      children.push_back(cond);
      children.push_back(execute);
      return children;
    }

  };

  /* Natural loop structure node */
  class NaturalLoopNode : public Node
  {
  public:
    NaturalLoopNode(const NodeList& children): Node(NaturalLoop, children){}
    virtual ~NaturalLoopNode() {}
    virtual ir::BasicBlock* getEntry()
    {
      //TODO implement it
      return NULL;
    }
    virtual ir::BasicBlock* getExit()
    {
      //TODO implement it
      return NULL;
    }
  };
#endif
  /* computes the control tree, and do the structure transform during the computation */
  class ControlTree
  {
  public:
    void analyze();

    ControlTree(ir::Function* fn) { this->fn = fn; }
    ~ControlTree();

  private:
    void initializeNodes();
    Node* insertNode(Node *);
    void structuralAnalysis(Node * entry);
    void DFSPostOrder(Node *start);
    bool path(Node *, Node *, Node *notthrough = NULL);
    void reduce(Node* node,  NodeSet nodeSet);
    void compact(Node* node,  NodeSet nodeSet);
    Node* getNodesEntry() const  { return nodes_entry;}
    Node* acyclicRegionType(Node*, NodeSet&);
    Node* cyclicRegionType(Node*, NodeList&);
    bool isCyclic(Node*);
    bool isBackedge(const Node*, const Node*);
    bool pathBack(Node*, Node*);
    bool checkForBarrier(const ir::BasicBlock*);
    void markStructuredNodes(Node *, bool);
    void markNeedEndif(Node *, bool);
    void markNeedIf(Node *, bool);
    void handleIfNode(Node *, ir::LabelIndex&, ir::LabelIndex&);
    void handleThenNode(Node *, ir::LabelIndex&);
    void handleThenNode2(Node *, Node *, ir::LabelIndex);
    void handleElseNode(Node *, ir::LabelIndex&, ir::LabelIndex&);
    void handleStructuredNodes();
    std::set<int> getStructureBasicBlocksIndex(Node *, std::vector<ir::BasicBlock *> &);
    std::set<ir::BasicBlock *> getStructureBasicBlocks(Node*);
    void getLiveIn(ir::BasicBlock& , std::set<ir::Register>& livein);
    void calculateNecessaryLiveout();
    void getStructureSequence(Node*, std::vector<ir::BasicBlock*> &);
  private:
    ir::Function *fn;
    NodeVector nodes;
    Node* nodes_entry;
    unordered_map<ir::BasicBlock *, Node *> bbmap;
    NodeList post_order;
    NodeSet visited;
    NodeList::iterator post_ctr;
  };
}
#endif
