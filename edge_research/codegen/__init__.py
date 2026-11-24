"""M10: MQL5 code generator and parity harness (design chapter 16).

``plan.build_plan``      frozen rule + config -> ``EAPlan`` (or ``CodegenRefusal``: an atom that cannot be translated exactly is never approximated).
``render.render_ea``     ``EAPlan`` -> MQL5 source text.
``pyeval``               evaluates the plan's expressions on a bar table: the feature / signal reference for the parity checks.
``ea_sim``               bar-by-bar model of the generated EA's OnTick algorithm (no MetaTrader in the sandbox): compared with the engine.
``parity``               feature / signal / trade parity checks of an EA dump and an EA trade log against the Python side (gate G9).
"""

CODEGEN_VERSION = "1.0"


class CodegenRefusal(Exception):
    """The generator refuses to produce an EA (design 16.1). ``reasons`` lists every cause found, not just the first."""

    def __init__(self, reasons):
        self.reasons = [reasons] if isinstance(reasons, str) else list(reasons)
        super().__init__("; ".join(self.reasons))
