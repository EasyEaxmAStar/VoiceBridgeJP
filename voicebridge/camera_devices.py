"""Native DirectShow identity enumeration; no camera is opened by listing."""
from __future__ import annotations

import ctypes as C
from ctypes import wintypes as W
from dataclasses import dataclass, asdict
import os
import uuid


@dataclass(frozen=True)
class CameraDevice:
    index: int
    name: str
    identity: str
    virtual: bool


def enumerate_cameras():
    if os.name != 'nt':raise RuntimeError('Webcam enumeration requires native Windows')
    ole=C.WinDLL('ole32');aut=C.OleDLL('oleaut32')
    class GUID(C.Structure):_fields_=[('data',C.c_ubyte*16)]
    def guid(text):return GUID((C.c_ubyte*16).from_buffer_copy(uuid.UUID(text).bytes_le))
    def call(ptr,index,types,*args):
        table=C.cast(ptr,C.POINTER(C.POINTER(C.c_void_p))).contents
        return C.WINFUNCTYPE(C.c_long,C.c_void_p,*types)(table[index])(ptr,*args)
    def release(ptr):
        if ptr:call(ptr,2,[])
    class Pair(C.Structure):_fields_=[('a',C.c_void_p),('b',C.c_void_p)]
    class Value(C.Union):_fields_=[('bstr',C.c_void_p),('number',C.c_double),('record',Pair)]
    class Variant(C.Structure):_fields_=[('vt',W.WORD),('r1',W.WORD),('r2',W.WORD),('r3',W.WORD),('value',Value)]
    ole.CoInitializeEx.argtypes=[C.c_void_p,W.DWORD]
    ole.CoInitializeEx.restype=C.c_long
    ole.CoCreateInstance.argtypes=[C.POINTER(GUID),C.c_void_p,W.DWORD,C.POINTER(GUID),C.POINTER(C.c_void_p)]
    ole.CoTaskMemFree.argtypes=[C.c_void_p]
    aut.VariantClear.argtypes=[C.POINTER(Variant)]
    result=ole.CoInitializeEx(None,2)
    # OpenCV may already have initialized this capture thread in another COM
    # apartment. Reuse it, without uninitializing a COM reference we don't own.
    if result not in (0,1,-2147417850):raise RuntimeError('DirectShow COM initialization failed')
    dev=enum=moniker=bag=C.c_void_p()
    devices=[]
    try:
        dev=C.c_void_p();enum=C.c_void_p()
        cls=guid('62BE5D10-60EB-11D0-BD3B-00A0C911CE86');iid=guid('29840822-5B84-11D0-BD3B-00A0C911CE86')
        if ole.CoCreateInstance(C.byref(cls),None,1,C.byref(iid),C.byref(dev))!=0:
            raise RuntimeError('DirectShow device enumerator unavailable')
        category=guid('860BB310-5D01-11D0-BD3B-00A0C911CE86')
        if call(dev,3,[C.POINTER(GUID),C.POINTER(C.c_void_p),W.DWORD],C.byref(category),C.byref(enum),0)!=0:return []
        while True:
            moniker=C.c_void_p();bag=C.c_void_p();fetched=W.ULONG()
            if call(enum,3,[W.ULONG,C.POINTER(C.c_void_p),C.POINTER(W.ULONG)],1,C.byref(moniker),C.byref(fetched))!=0:break
            try:
                iid=guid('55272A00-42CB-11CE-8135-00AA004BB851')
                hr=call(moniker,9,[C.c_void_p,C.c_void_p,C.POINTER(GUID),C.POINTER(C.c_void_p)],None,None,C.byref(iid),C.byref(bag))
                if hr:raise RuntimeError('Cannot read DirectShow camera properties')
                value=Variant()
                try:
                    if call(bag,3,[W.LPCWSTR,C.POINTER(Variant),C.c_void_p],'FriendlyName',C.byref(value),None)!=0 or value.vt!=8:
                        raise RuntimeError('Camera lacks a valid FriendlyName')
                    name=C.wstring_at(value.value.bstr)
                finally:aut.VariantClear(C.byref(value))
                display=C.c_void_p()
                if call(moniker,20,[C.c_void_p,C.c_void_p,C.POINTER(C.c_void_p)],None,None,C.byref(display))!=0:
                    raise RuntimeError('Camera lacks a stable DirectShow identity')
                try:identity=C.wstring_at(display)
                finally:ole.CoTaskMemFree(display)
                virtual=identity.casefold().startswith('@device:sw:') or any(token in name.casefold() for token in ('virtual','unity video','obs camera'))
                devices.append(CameraDevice(len(devices),name,identity,virtual))
            finally:release(bag);bag=C.c_void_p();release(moniker);moniker=C.c_void_p()
        return devices
    finally:
        release(enum);release(dev)
        if result in (0,1):ole.CoUninitialize()


def resolve_camera(identity, devices=None):
    matches=[d for d in (enumerate_cameras() if devices is None else devices) if d.identity==identity and not d.virtual]
    if len(matches)!=1:raise RuntimeError('Selected physical camera is missing/ambiguous. Re-run --diagnose; no alternate camera was selected.')
    return matches[0]


def diagnostics():
    cameras=enumerate_cameras()
    return dict(cameras=[asdict(d) for d in cameras],
                virtual_backends=[dict(backend='obs',device=d.name) for d in cameras if d.name=='OBS Virtual Camera']+
                [dict(backend='unitycapture',device=d.name) for d in cameras if 'unity' in d.name.casefold()],
                note='Enumeration only. Backend ownership, formats and permissions require an explicit smoke test.')
