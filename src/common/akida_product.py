"""Product names (AKD1000 / AKD1500) from what the runtime and the launcher report.

Pure stdlib, and deliberately no ``import akida``: the same mapping is needed on the host
(the three dashboards, which never touch a chip and run under the repo's uv venv) and inside
the container (akida_chip, which does). One table, both sides.

The runtime never hands out the product name. It offers two strings, and neither is what a
demo should put on screen:

    device.version -> "BC.A1.003.009"     the HwVersion; exact, and what we key on
    device.desc    -> "PCIe/NSoC_v2/0"    the SILICON name -- AKD1000 is sold as such

so NSoC_v2 has to be translated, and the akida SDK agrees: akida.AKD1000().version IS
akida.NSoC_v2. desc also carries no per-node identity here, because docker/entrypoint.sh
remaps each container's assigned chip to slot 0 of its family -- every node reports
".../0". The physical chip a node owns comes from AKIDA_CHIP_NODE (from_chip_node below),
never from desc.

Every lookup ends in a non-empty string. That is a contract, not politeness: the
serial-http dashboard counts truthy "hardware" fields to report "all N ON-CHIP", so an
empty product name would silently turn a fully on-chip run into "0/N on-chip".
"""

# HwVersion string -> product name. The comment on each line is the akida constant it
# equals, so a new SDK can be diffed against `[n for n in dir(akida) if isinstance(
# getattr(akida, n), akida.HwVersion)]` -- that is where these six came from (akida 2.19.2).
PRODUCTS = {
    "BC.00.000.001": "AKD1000",       # akida.NSoC_v1     -- pre-production v1 silicon
    "BC.00.000.002": "AKD1000",       # akida.NSoC_v2     == akida.AKD1000().version
    "BC.A1.003.009": "AKD1500",       # akida.AKD1500_v1  == akida.AKD1500().version
    "BC.A1.003.006": "TwoNodesIP",    # akida.TwoNodesIP_v1
    "BC.A2.001.000": "TwoNodesIPv2",  # akida.FPGA_v2     == akida.TwoNodesIPv2().version
    "BC.B1.001.000": "Akida Pico",    # akida.Pico_FPGA
}

# Silicon name (desc segment 1) -> product name, for the descs whose silicon name is not
# already the product name. Only the NSoC line needs it; AKD1500 names itself.
SILICON = {
    "NSoC_v1": "AKD1000",
    "NSoC_v2": "AKD1000",
}

UNKNOWN = "Akida"


def from_version(version):
    """Product name for a HwVersion string ("BC.A1.003.009"), or None if unrecognised."""
    return PRODUCTS.get(str(version or "").strip())


def from_desc(desc):
    """Product name from a device desc. Never empty.

    Three desc shapes exist and all three reach here:

        PCIe/AKD1500/16MB/0            a real AKD1500 (4 segments)
        PCIe/NSoC_v2/0                 a real AKD1000 (3 segments)
        VirtualDevice/BC.A1.003.009    akida.AKD1500() and friends -- segment 1 is a version

    So take segment 1, hand it back to the version table when it looks like a HwVersion,
    then normalise the silicon name. An unrecognised desc degrades to its own silicon
    segment rather than to nothing, which keeps a future family legible instead of blank.
    """
    desc = str(desc or "").strip()
    if not desc:
        return UNKNOWN
    parts = desc.split("/")
    silicon = parts[1].strip() if len(parts) > 1 else desc
    return (from_version(silicon) or SILICON.get(silicon) or silicon or desc or UNKNOWN)


def from_chip_node(node):
    """Product name for a /dev node basename ("akd1500_3", "akida0"), or None.

    The launcher's own family split (probe_chips.sh: AKD1500 is PCI 1e7c:a500 -> akd1500_<N>,
    AKD1000 is 1e7c:bca1 -> akida<N>). This is the chip a node was ASSIGNED, which is all the
    host can know before a worker has opened the device -- so callers should label it as such
    rather than as the chip's own claim.
    """
    node = str(node or "").strip()
    if node.startswith("akd1500"):
        return "AKD1500"
    if node.startswith("akida"):
        return "AKD1000"
    return None
