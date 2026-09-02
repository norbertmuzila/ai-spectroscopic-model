"""
Hardware diagnostics.

"The spectrometer isn't reading" has half a dozen distinct causes, and they need
completely different fixes. This module distinguishes them and returns a verdict
plus the specific next step, rather than a generic failure.

The checks run outermost-inward, because a failure at any level makes everything
below it meaningless:

    1. Is the seabreeze library importable, and which backends built?
    2. Is a USB backend (libusb) present for the pure-Python backend?
    3. Does the OS see an Ocean Optics device on the USB bus *right now*?
    4. Does Windows have a driver bound to it?
    5. Can seabreeze actually open and read it?

Step 3 is the one that catches the most common real-world cause. Windows keeps a
record of every device ever attached, so Device Manager and most tooling will
happily show a USB4000 that was unplugged weeks ago. Those "ghost" entries make
people conclude the driver is broken when the cable is simply not connected, so
the presence check here is explicitly *present-only* and ghosts are reported
separately and labelled as such.
"""
from __future__ import annotations

import platform
import subprocess
import sys

OCEAN_VID = 0x2457
# Ocean Optics product IDs, so a device that is present but not a USB4000 is
# named rather than reported as "unknown".
OCEAN_PIDS = {
    0x1002: "USB2000", 0x100a: "HR2000", 0x1012: "HR4000", 0x1014: "HR2000+",
    0x1016: "QE65000", 0x1022: "USB4000", 0x1024: "NIRQuest512",
    0x1026: "NIRQuest256", 0x1028: "Maya2000Pro", 0x102a: "Maya2000",
    0x1030: "Torus", 0x1032: "Apex", 0x1038: "Maya LSL", 0x1040: "Jaz",
    0x1044: "STS", 0x1046: "QE-Pro", 0x1048: "Ventana", 0x2000: "Spark",
    0x4000: "USB2000+", 0x4004: "Flame-S", 0x2001: "Flame-NIR",
}


def _run(cmd: list, timeout: float = 25.0) -> str:
    try:
        r = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
        return (r.stdout or "") + (r.stderr or "")
    except Exception as exc:
        return f"<failed: {exc}>"


# ---------------------------------------------------------------------------
def check_library() -> dict:
    out = {"seabreeze_installed": False, "version": None, "backends": {}}
    try:
        import seabreeze
        out["seabreeze_installed"] = True
        out["version"] = getattr(seabreeze, "__version__", "?")
    except Exception as exc:
        out["error"] = f"{type(exc).__name__}: {exc}"
        return out

    for backend in ("cseabreeze", "pyseabreeze"):
        try:
            mod = __import__(f"seabreeze.{backend}", fromlist=["*"])
            out["backends"][backend] = {"available": True,
                                        "module": getattr(mod, "__name__", backend)}
        except Exception as exc:
            out["backends"][backend] = {"available": False,
                                        "error": f"{type(exc).__name__}: {exc}"[:200]}
    return out


def check_usb_backend() -> dict:
    out = {"pyusb": False, "libusb": False, "n_usb_devices": None}
    try:
        import usb.core
        out["pyusb"] = True
    except Exception as exc:
        out["error"] = f"pyusb missing: {exc}"
        return out

    backend = None
    try:
        import libusb_package
        backend = libusb_package.get_libusb1_backend()
        out["libusb_source"] = "libusb-package"
    except Exception:
        try:
            import usb.backend.libusb1 as l1
            backend = l1.get_backend()
            out["libusb_source"] = "system libusb-1.0"
        except Exception as exc:
            out["error"] = f"no libusb backend: {exc}"

    out["libusb"] = backend is not None
    if backend is not None:
        try:
            out["n_usb_devices"] = len(list(usb.core.find(find_all=True, backend=backend)))
        except Exception as exc:
            out["error"] = f"bus scan failed: {exc}"
    return out


def scan_usb_bus() -> dict:
    """Ocean Optics devices physically on the bus right now."""
    out = {"scanned": False, "devices": []}
    try:
        import usb.core
        backend = None
        try:
            import libusb_package
            backend = libusb_package.get_libusb1_backend()
        except Exception:
            pass
        devs = usb.core.find(find_all=True, idVendor=OCEAN_VID, backend=backend)
        out["scanned"] = True
        for d in devs:
            out["devices"].append({
                "product_id": f"0x{d.idProduct:04x}",
                "model": OCEAN_PIDS.get(d.idProduct, "unknown Ocean Optics model"),
                "bus": getattr(d, "bus", None),
                "address": getattr(d, "address", None),
            })
    except Exception as exc:
        out["error"] = f"{type(exc).__name__}: {exc}"
    return out


def scan_windows_pnp() -> dict:
    """
    Ask Windows what it sees, separating attached devices from remembered ones.

    The ghost/present split is the whole point: a remembered entry looks
    identical to a live one in Device Manager, and mistaking one for the other
    sends people driver-hunting when the real problem is a cable.
    """
    out = {"platform": platform.system(), "present": [], "ghost": [], "checked": False}
    if platform.system() != "Windows":
        return out

    ps = (
        "$ErrorActionPreference='SilentlyContinue';"
        "$all = Get-PnpDevice | Where-Object { $_.InstanceId -like '*VID_2457*' };"
        "$present = Get-PnpDevice -PresentOnly | Where-Object { $_.InstanceId -like '*VID_2457*' };"
        "$pi = @($present | ForEach-Object { $_.InstanceId });"
        "foreach ($d in $all) {"
        "  $isPresent = $pi -contains $d.InstanceId;"
        "  $svc = '';"
        "  try { $svc = (Get-PnpDeviceProperty -InstanceId $d.InstanceId "
        "        -KeyName 'DEVPKEY_Device_Service').Data } catch {}"
        "  Write-Output ((@($(if($isPresent){'PRESENT'}else{'GHOST'}), $d.FriendlyName, "
        "        $d.Status, $svc, $d.InstanceId, $d.Problem, $d.ProblemDescription)) "
        "        -join '|') }"
    )
    text = _run(["powershell", "-NoProfile", "-NonInteractive", "-Command", ps])
    out["checked"] = "<failed" not in text
    for line in text.splitlines():
        parts = line.strip().split("|")
        if len(parts) < 5 or parts[0] not in ("PRESENT", "GHOST"):
            continue
        rec = {
            "name": parts[1], "status": parts[2],
            "driver_service": parts[3] or None,
            "instance": parts[4],
            "problem_code": parts[5] if len(parts) > 5 else "",
            "problem": parts[6] if len(parts) > 6 else "",
        }
        (out["present"] if parts[0] == "PRESENT" else out["ghost"]).append(rec)
    return out


def _dashboard_holds_device(port: int = 8000, timeout: float = 2.0) -> bool:
    """Is the local dashboard running and currently connected to real hardware?"""
    import json
    import urllib.request
    try:
        with urllib.request.urlopen(
                f"http://127.0.0.1:{port}/api/status", timeout=timeout) as r:
            st = json.loads(r.read().decode("utf-8"))
        return bool(st.get("connected")) and not st.get("device", {}).get("simulated", True)
    except Exception:
        return False


def try_open() -> dict:
    """Ask seabreeze itself to enumerate and, if possible, read one spectrum."""
    out = {"listed": [], "opened": None}
    try:
        from seabreeze.spectrometers import Spectrometer, list_devices
        devices = list_devices()
        out["listed"] = [{"model": d.model, "serial": d.serial_number} for d in devices]
        if not devices:
            return out
        spec = Spectrometer(devices[0])
        wl = spec.wavelengths()
        inten = spec.intensities()
        out["opened"] = {
            "model": spec.model,
            "serial": spec.serial_number,
            "pixels": int(len(wl)),
            "wavelength_min_nm": round(float(wl.min()), 2),
            "wavelength_max_nm": round(float(wl.max()), 2),
            "peak_counts": round(float(inten.max()), 1),
        }
        spec.close()
    except Exception as exc:
        out["error"] = f"{type(exc).__name__}: {exc}"[:300]
    return out


# ---------------------------------------------------------------------------
def diagnose() -> dict:
    lib = check_library()
    usb_be = check_usb_backend()
    bus = scan_usb_bus()
    pnp = scan_windows_pnp()
    opened = try_open() if lib.get("seabreeze_installed") else {"listed": [], "opened": None}

    on_bus = len(bus.get("devices", []))
    present = len(pnp.get("present", []))
    ghosts = len(pnp.get("ghost", []))
    listed = len(opened.get("listed", []))

    verdict, severity, steps = "", "error", []

    if not lib.get("seabreeze_installed"):
        verdict = "The seabreeze library is not installed, so nothing can talk to the spectrometer."
        steps = ['Run:  python -m pip install "seabreeze[pyseabreeze]" libusb-package']

    elif opened.get("opened"):
        d = opened["opened"]
        verdict = (f"{d['model']} (serial {d['serial']}) is connected and reading: "
                   f"{d['pixels']} pixels over {d['wavelength_min_nm']}-{d['wavelength_max_nm']} nm.")
        severity = "ok"
        steps = ["Press Connect on the dashboard - the hardware path is working."]

    elif on_bus == 0 and present == 0:
        verdict = ("No Ocean Optics device is attached to this computer right now. "
                   + (f"Windows remembers {ghosts} USB4000 connection"
                      f"{'s' if ghosts != 1 else ''} from earlier, but "
                      f"{'they are' if ghosts != 1 else 'it is'} not plugged in now."
                      if ghosts else ""))
        severity = "error"
        steps = [
            "Plug the USB4000 into a USB port directly on the computer, not through "
            "an unpowered hub - it draws around 450 mA and browns out on a weak port.",
            "Use a full data cable. A charge-only USB cable powers the lamp but carries "
            "no data, and looks identical.",
            "Watch for the Windows connection chime, then re-run this diagnostic.",
            "If it still does not appear, try a different USB port and cable - the "
            "'Device Descriptor Request Failed' entries in Device Manager are the "
            "signature of a failing cable or an underpowered port.",
        ]

    elif on_bus == 0 and present > 0:
        bound = [p for p in pnp["present"] if p.get("driver_service")]
        codes = {p.get("problem_code") for p in pnp["present"]}
        severity = "error"

        if bound:
            # A driver is attached, just not one libusb can use. Almost always
            # the Ocean Optics vendor driver installed by SpectraSuite.
            drv = bound[0]["driver_service"]
            verdict = (f"Windows has bound the '{drv}' driver to the spectrometer, but "
                       f"libusb cannot use it. seabreeze needs WinUSB, so the driver has "
                       f"to be replaced.")
            steps = [
                "Close SpectraSuite and OceanView first - they hold the device open.",
                "Install Zadig from https://zadig.akeo.ie/ , choose "
                "Options > List All Devices, select 'Ocean Optics USB4000' in the "
                "dropdown, pick WinUSB on the right, and click Replace Driver.",
                "Then re-run this diagnostic.",
            ]
        else:
            # CM_PROB_FAILED_INSTALL (code 28): enumerated, but no driver at all.
            # Get-PnpDevice reports the symbolic name, and older builds report the
            # number, so accept either rather than silently missing the case.
            code28 = any("FAILED_INSTALL" in c or c.strip() == "28"
                         for c in codes if c)
            verdict = ("Windows sees the spectrometer but has bound no driver to it"
                       + (" (problem code 28, CM_PROB_FAILED_INSTALL)" if code28 else "")
                       + ", so nothing can open it - not this software, not SpectraSuite.")
            steps = [
                "Run INSTALL-DRIVER.bat in the project folder and accept the Windows "
                "security prompt. It installs OOI_USB4000.inf, which matches "
                "USB\VID_2457&PID_1022 exactly and binds Microsoft's WinUSB driver.",
                "Installing a driver needs Administrator rights, which is why the "
                "dashboard cannot do it for you.",
                "If Windows rejects the package (its catalog is from 2010 and signed "
                "with SHA-1), use Zadig from https://zadig.akeo.ie/ instead: "
                "Options > List All Devices, select 'Ocean Optics USB4000', "
                "choose WinUSB, click Replace Driver.",
                "Close SpectraSuite and OceanView before measuring - only one process "
                "can hold the device.",
            ]

    elif on_bus > 0 and listed == 0:
        names = ", ".join(d["model"] for d in bus["devices"])
        # The most likely holder is this project's own dashboard. Only one
        # process can own a USB4000, so a connected console makes every other
        # process - including this script - see a device it cannot open. Saying
        # "another program is holding it" without naming that case sends people
        # hunting for software they already closed.
        dashboard = _dashboard_holds_device()
        severity = "error"
        if dashboard:
            verdict = (f"The spectrometer ({names}) is working, but the Spectral Console "
                       f"dashboard already has it open. Only one process can own a "
                       f"USB4000, so this separate check cannot also open it.")
            steps = [
                "Nothing is wrong - the dashboard is connected and using the device.",
                "To run this diagnostic against the hardware, press Stop on the "
                "dashboard first, or use the dashboard's own Hardware panel, which "
                "checks from inside the process that holds the device.",
            ]
        else:
            verdict = (f"The device is on the USB bus ({names}) but seabreeze cannot "
                       f"claim it. Another program is holding it open.")
            steps = [
                "Close SpectraSuite, OceanView, and the Spectral Console dashboard if "
                "it is running - any of them will hold the device.",
                "If nothing else is running, the bound driver is the vendor one rather "
                "than WinUSB - use Zadig to replace it.",
            ]

    else:
        verdict = ("The device is visible but could not be read. "
                   + str(opened.get("error", "")))
        severity = "warning"
        steps = ["Unplug the spectrometer, wait five seconds, plug it back in and retry.",
                 "Close any other Ocean Optics software that may hold the device."]

    if not usb_be.get("libusb"):
        steps.append("Install the USB backend as well:  python -m pip install libusb-package")

    return {
        "verdict": verdict,
        "severity": severity,
        "next_steps": steps,
        "hardware_ready": bool(opened.get("opened")),
        "summary": {
            "ocean_devices_on_bus": on_bus,
            "windows_present": present,
            "windows_ghost_entries": ghosts,
            "seabreeze_lists": listed,
            "usb_devices_total": usb_be.get("n_usb_devices"),
        },
        "library": lib,
        "usb_backend": usb_be,
        "bus_scan": bus,
        "windows_pnp": pnp,
        "open_attempt": opened,
        "python": sys.version.split()[0],
        "platform": f"{platform.system()} {platform.release()}",
    }
