from amaranth import *
from amaranth.sim import *
from amaranth.back import verilog
from toycore.utils import *
import unittest

class Counter4Bit(Elaboratable):
    def __init__(self):
        self.enable = Signal()
        self.count  = Signal(4)

    def elaborate(self, platform):
        m = Module()

        with m.If(self.enable):
            m.d.sync += self.count.eq(self.count + 1)

        return m
    
class TestCounter(unittest.TestCase):
    def test_counter_behavior(self):
        dut = Counter4Bit()

        def bench():
            # Test reset
            yield dut.enable.eq(0)
            yield Tick()
            assert (yield dut.count) == 0

            # Test counting
            dut.enable.eq(1)
            for i in range(20):
                yield Tick()
                assert (yield dut.count) == (i + 1) % 16, f"Expected {(i + 1) % 16}, got {yield dut.count}"

            # Test disable
            yield dut.enable.eq(0)
            yield Tick()
            count_after_disable = yield dut.count
            yield Tick()
            assert (yield dut.count) == count_after_disable, "Counter should not change when disabled"

            # Test enable again
            yield dut.enable.eq(1)
            prev_count = yield dut.count
            yield Tick()
            assert (yield dut.count) == (prev_count + 1) % 16, "Counter should resume counting when re-enabled"

        sim = Simulator(dut)
        sim.add_clock(1e-6)
        sim.add_process(bench)
        with sim.write_vcd(str(get_or_create_tmp() / "waveform.vcd")):
            sim.run()

if __name__ == '__main__':
    unittest.main()

    # Generate Verilog
    dut = Counter4Bit()
    with open(get_or_create_tmp() / "counter_4bit.v", "w") as f:
        f.write(verilog.convert(dut, ports=[dut.enable, dut.count]))
    print("Verilog file 'counter_4bit.v' generated.")