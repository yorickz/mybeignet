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

/*
 * Copyright � 2008 Red Hat, Inc.
 *
 * Permission is hereby granted, free of charge, to any person obtaining a
 * copy of this software and associated documentation files (the "Soft-
 * ware"), to deal in the Software without restriction, including without
 * limitation the rights to use, copy, modify, merge, publish, distribute,
 * and/or sell copies of the Software, and to permit persons to whom the
 * Software is furnished to do so, provided that the above copyright
 * notice(s) and this permission notice appear in all copies of the Soft-
 * ware and that both the above copyright notice(s) and this permission
 * notice appear in supporting documentation.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS
 * OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABIL-
 * ITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT OF THIRD PARTY
 * RIGHTS. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR HOLDERS INCLUDED IN
 * THIS NOTICE BE LIABLE FOR ANY CLAIM, OR ANY SPECIAL INDIRECT OR CONSE-
 * QUENTIAL DAMAGES, OR ANY DAMAGES WHATSOEVER RESULTING FROM LOSS OF USE,
 * DATA OR PROFITS, WHETHER IN AN ACTION OF CONTRACT, NEGLIGENCE OR OTHER
 * TORTIOUS ACTION, ARISING OUT OF OR IN CONNECTION WITH THE USE OR PERFOR-
 * MANCE OF THIS SOFTWARE.
 *
 * Except as contained in this notice, the name of a copyright holder shall
 * not be used in advertising or otherwise to promote the sale, use or
 * other dealings in this Software without prior written authorization of
 * the copyright holder.
 *
 * Authors:
 *   Kristian H�gsberg (krh@redhat.com)
 */

#define NEED_REPLIES
#include <X11/Xlibint.h>
#include <X11/extensions/Xext.h>
#include <X11/extensions/extutil.h>
#include "xf86drm.h"
#include "x11/va_dri2.h"
#include "x11/va_dri2str.h"
#include "x11/va_dri2tokens.h"

#ifndef DRI2DriverDRI
#define DRI2DriverDRI 0
#endif

#define LOCAL __attribute__ ((visibility ("internal")))

static char va_dri2ExtensionName[] = DRI2_NAME;
static XExtensionInfo _va_dri2_info_data;
static XExtensionInfo *va_dri2Info = &_va_dri2_info_data;
static XEXT_GENERATE_CLOSE_DISPLAY (VA_DRI2CloseDisplay, va_dri2Info)
static /* const */ XExtensionHooks va_dri2ExtensionHooks = {
    NULL,				/* create_gc */
    NULL,				/* copy_gc */
    NULL,				/* flush_gc */
    NULL,				/* free_gc */
    NULL,				/* create_font */
    NULL,				/* free_font */
    VA_DRI2CloseDisplay,		/* close_display */
    NULL,				/* wire_to_event */
    NULL,				/* event_to_wire */
    NULL,				/* error */
    NULL,				/* error_string */
};

static XEXT_GENERATE_FIND_DISPLAY (DRI2FindDisplay, va_dri2Info, 
				   va_dri2ExtensionName, 
				   &va_dri2ExtensionHooks, 
				   0, NULL)

LOCAL Bool VA_DRI2QueryExtension(Display *dpy, int *eventBase, int *errorBase)
{
    XExtDisplayInfo *info = DRI2FindDisplay(dpy);

    if (XextHasExtension(info)) {
	*eventBase = info->codes->first_event;
	*errorBase = info->codes->first_error;
	return True;
    }

    return False;
}

LOCAL Bool VA_DRI2QueryVersion(Display *dpy, int *major, int *minor)
{
    XExtDisplayInfo *info = DRI2FindDisplay (dpy);
    xDRI2QueryVersionReply rep;
    xDRI2QueryVersionReq *req;

    XextCheckExtension (dpy, info, va_dri2ExtensionName, False);

    LockDisplay(dpy);
    GetReq(DRI2QueryVersion, req);
    req->reqType = info->codes->major_opcode;
    req->dri2Reqtype = X_DRI2QueryVersion;
    req->majorVersion = DRI2_MAJOR;
    req->minorVersion = DRI2_MINOR;
    if (!_XReply(dpy, (xReply *)&rep, 0, xFalse)) {
	UnlockDisplay(dpy);
	SyncHandle();
	return False;
    }
    *major = rep.majorVersion;
    *minor = rep.minorVersion;
    UnlockDisplay(dpy);
    SyncHandle();

    return True;
}

LOCAL Bool VA_DRI2Connect(Display *dpy, XID window,
		 char **driverName, char **deviceName)
{
    XExtDisplayInfo *info = DRI2FindDisplay(dpy);
    xDRI2ConnectReply rep;
    xDRI2ConnectReq *req;

    XextCheckExtension (dpy, info, va_dri2ExtensionName, False);

    LockDisplay(dpy);
    GetReq(DRI2Connect, req);
    req->reqType = info->codes->major_opcode;
    req->dri2Reqtype = X_DRI2Connect;
    req->window = window;
    req->drivertype = DRI2DriverDRI;
    if (!_XReply(dpy, (xReply *)&rep, 0, xFalse)) {
	UnlockDisplay(dpy);
	SyncHandle();
	return False;
    }

    if (rep.driverNameLength == 0 && rep.deviceNameLength == 0) {
	UnlockDisplay(dpy);
	SyncHandle();
	return False;
    }

    *driverName = Xmalloc(rep.driverNameLength + 1);
    if (*driverName == NULL) {
	_XEatData(dpy, 
		  ((rep.driverNameLength + 3) & ~3) +
		  ((rep.deviceNameLength + 3) & ~3));
	UnlockDisplay(dpy);
	SyncHandle();
	return False;
    }
    _XReadPad(dpy, *driverName, rep.driverNameLength);
    (*driverName)[rep.driverNameLength] = '\0';

    *deviceName = Xmalloc(rep.deviceNameLength + 1);
    if (*deviceName == NULL) {
	Xfree(*driverName);
	_XEatData(dpy, ((rep.deviceNameLength + 3) & ~3));
	UnlockDisplay(dpy);
	SyncHandle();
	return False;
    }
    _XReadPad(dpy, *deviceName, rep.deviceNameLength);
    (*deviceName)[rep.deviceNameLength] = '\0';

    UnlockDisplay(dpy);
    SyncHandle();

    return True;
}

LOCAL Bool VA_DRI2Authenticate(Display *dpy, XID window, drm_magic_t magic)
{
    XExtDisplayInfo *info = DRI2FindDisplay(dpy);
    xDRI2AuthenticateReq *req;
    xDRI2AuthenticateReply rep;

    XextCheckExtension (dpy, info, va_dri2ExtensionName, False);

    LockDisplay(dpy);
    GetReq(DRI2Authenticate, req);
    req->reqType = info->codes->major_opcode;
    req->dri2Reqtype = X_DRI2Authenticate;
    req->window = window;
    req->magic = magic;

    if (!_XReply(dpy, (xReply *)&rep, 0, xFalse)) {
	UnlockDisplay(dpy);
	SyncHandle();
	return False;
    }

    UnlockDisplay(dpy);
    SyncHandle();

    return rep.authenticated;
}

LOCAL void VA_DRI2CreateDrawable(Display *dpy, XID drawable)
{
    XExtDisplayInfo *info = DRI2FindDisplay(dpy);
    xDRI2CreateDrawableReq *req;

    XextSimpleCheckExtension (dpy, info, va_dri2ExtensionName);

    LockDisplay(dpy);
    GetReq(DRI2CreateDrawable, req);
    req->reqType = info->codes->major_opcode;
    req->dri2Reqtype = X_DRI2CreateDrawable;
    req->drawable = drawable;
    UnlockDisplay(dpy);
    SyncHandle();
}

LOCAL void VA_DRI2DestroyDrawable(Display *dpy, XID drawable)
{
    XExtDisplayInfo *info = DRI2FindDisplay(dpy);
    xDRI2DestroyDrawableReq *req;

    XextSimpleCheckExtension (dpy, info, va_dri2ExtensionName);

    XSync(dpy, False);

    LockDisplay(dpy);
    GetReq(DRI2DestroyDrawable, req);
    req->reqType = info->codes->major_opcode;
    req->dri2Reqtype = X_DRI2DestroyDrawable;
    req->drawable = drawable;
    UnlockDisplay(dpy);
    SyncHandle();
}

LOCAL VA_DRI2Buffer *VA_DRI2GetBuffers(Display *dpy, XID drawable,
			   int *width, int *height,
			   unsigned int *attachments, int count,
			   int *outcount)
{
    XExtDisplayInfo *info = DRI2FindDisplay(dpy);
    xDRI2GetBuffersReply rep;
    xDRI2GetBuffersReq *req;
    VA_DRI2Buffer *buffers;
    xDRI2Buffer repBuffer;
    CARD32 *p;
    int i;

    XextCheckExtension (dpy, info, va_dri2ExtensionName, False);

    LockDisplay(dpy);
    GetReqExtra(DRI2GetBuffers, count * 4, req);
    req->reqType = info->codes->major_opcode;
    req->dri2Reqtype = X_DRI2GetBuffers;
    req->drawable = drawable;
    req->count = count;
    p = (CARD32 *) &req[1];
    for (i = 0; i < count; i++)
	p[i] = attachments[i];

    if (!_XReply(dpy, (xReply *)&rep, 0, xFalse)) {
	UnlockDisplay(dpy);
	SyncHandle();
	return NULL;
    }

    *width = rep.width;
    *height = rep.height;
    *outcount = rep.count;

    buffers = Xmalloc(rep.count * sizeof buffers[0]);
    if (buffers == NULL) {
	_XEatData(dpy, rep.count * sizeof repBuffer);
	UnlockDisplay(dpy);
	SyncHandle();
	return NULL;
    }

    for (i = 0; i < (int) rep.count; i++) {
	_XReadPad(dpy, (char *) &repBuffer, sizeof repBuffer);
	buffers[i].attachment = repBuffer.attachment;
	buffers[i].name = repBuffer.name;
	buffers[i].pitch = repBuffer.pitch;
	buffers[i].cpp = repBuffer.cpp;
	buffers[i].flags = repBuffer.flags;
    }

    UnlockDisplay(dpy);
    SyncHandle();

    return buffers;
}

LOCAL void VA_DRI2CopyRegion(Display *dpy, XID drawable, XserverRegion region,
		    CARD32 dest, CARD32 src)
{
    XExtDisplayInfo *info = DRI2FindDisplay(dpy);
    xDRI2CopyRegionReq *req;
    xDRI2CopyRegionReply rep;

    XextSimpleCheckExtension (dpy, info, va_dri2ExtensionName);

    LockDisplay(dpy);
    GetReq(DRI2CopyRegion, req);
    req->reqType = info->codes->major_opcode;
    req->dri2Reqtype = X_DRI2CopyRegion;
    req->drawable = drawable;
    req->region = region;
    req->dest = dest;
    req->src = src;

    _XReply(dpy, (xReply *)&rep, 0, xFalse);

    UnlockDisplay(dpy);
    SyncHandle();
}
