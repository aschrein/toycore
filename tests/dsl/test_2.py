"""
    Here we're gonna try to implement 1.58 bit systolic array.
    The systolic array is a 1.58 bit adder that can add/subtract two numbers.


    Building Block:
    ---------------
        * 2x FlipFlop2Bit: A flip-flop is a memory element that stores a single bit of information with sign bit.
        * Accumulator32bit: A 32-bit accumulator.
        * Adder2Bit: A signed 2-bit adder.
    
    signed 2bit adder is just a 2bit adder with twos complement inversion block.

    The ring of 2 bit numbers is such:

                00
              /    \
            11      01
              \    /
                10 - invalid

        00 - 0
        01 - 1
        10 - -2 - invalid
        11 - -1

        truth table of 1.58 bit multiplier:

        | a_1 | a_0 | b_1 | b_0 | out_0 | out_1 |
        |-----|-----|-----|-----|-------|-------|
        |  0  |  0  |  0  |  0  |   0   |   0   | -> 0 * 0   = 0
        |  0  |  0  |  0  |  1  |   0   |   0   | -> 0 * 1   = 0
        |  0  |  0  |  1  |  0  |   0   |   0   | -> 0 * -2  = invalid
        |  0  |  0  |  1  |  1  |   0   |   0   | -> 0 * -1  = 0
        |  0  |  1  |  0  |  0  |   0   |   0   | -> 1 * 0   = 0
        |  0  |  1  |  0  |  1  |   0   |   0   | -> 1 * 1   = 1
        |  0  |  1  |  1  |  0  |   0   |   0   | -> 1 * -2  = invalid
        |  0  |  1  |  1  |  1  |   0   |   0   | -> 1 * -1  = -1
        |  1  |  0  |  0  |  0  |   0   |   0   | -> -2 * 0  = invalid
        |  1  |  0  |  0  |  1  |   0   |   0   | -> -2 * 1  = invalid
        |  1  |  0  |  1  |  0  |   0   |   0   | -> -2 * -2 = invalid
        |  1  |  0  |  1  |  1  |   0   |   0   | -> -2 * -1 = invalid
        |  1  |  1  |  0  |  0  |   0   |   0   | -> -1 * 0  = 0
        |  1  |  1  |  0  |  1  |   0   |   0   | -> -1 * 1  = -1
        |  1  |  1  |  1  |  0  |   0   |   0   | -> -1 * -2 = invalid
        |  1  |  1  |  1  |  1  |   0   |   0   | -> -1 * -1 = 1

        the boolean formula for 2bit multiplier is:
        out_0 = a_0 AND b_0
        out_1 = ((~a_1 AND a_0) AND (b_0 AND b_1)) OR ((a_1 AND a_0) AND (b_0 AND ~b_1))

"""

class DesignContext:
    """
        Design context is a container for all the elements of the design
        It is a singleton class, and the current context can be accessed using
        DesignContext.get_current()
    """
    
    # add static field current
    current = None

    def __init__(self):
        self.wires          = []
        self.gates          = []
        self.slots          = []
        self.logic_elements = []
        self.flip_flops     = []

    def add_flip_flop(self, flip_flop):
        self.flip_flops.append(flip_flop)
        return flip_flop

    def add_wire(self, wire):
        self.wires.append(wire)
        self.logic_elements.append(wire)

    def add_gate(self, gate):
        self.gates.append(gate)
        self.logic_elements.append(gate)
        return gate

    def add_slot(self, slot):
        self.slots.append(slot)
        self.logic_elements.append(slot)
        return slot

    def update(self):
        # Rising edge of the global clock triggers the flip-flops to update their state
        # and the gates to update their output 
        for flip_flop in self.flip_flops:
            flip_flop.update()

        topological_order = []
        visited = set()
        def dfs(node):
            if node in visited:
                return
            visited.add(node)
            for wire in node.inputs:
                dfs(wire)
            topological_order.append(node)
        
        for element in self.logic_elements:
            dfs(element)
        
        for element in topological_order:
            element.update()

    @staticmethod
    def clean():
        DesignContext.current = DesignContext()

    # Get current context
    @staticmethod
    def get_current():
        return DesignContext.current
    
class Slot1Bit:
    def __init__(self, parent, value=0):
        self.value          = value
        self.parent         = parent
        self.in_wires       = []
        self.out_wires      = []
        DesignContext.get_current().add_slot(self)
    
    @property
    def inputs(self): return self.in_wires
    @property
    def outputs(self): return self.out_wires

    def update(self):
        """
            Do nothing, slots are driven by wires or gates
        """

class Wire1Bit:
    def __init__(self, origin: Slot1Bit, end: Slot1Bit):
        self.origin = origin
        self.end    = end
        self.origin.out_wires.append(self)
        self.end.in_wires.append(self)
        assert len(self.end.in_wires) == 1, f"Slot {self.end} has multiple inputs"
        DesignContext.get_current().add_wire(self)

    @property
    def inputs(self): return [self.origin]
    @property
    def outputs(self): return [self.end]

    def update(self):
        self.end.value = self.origin.value

class UnaryGate1Bit:
    def __init__(self, a : Slot1Bit, op):
        self.in_0       = Slot1Bit(self)
        self.out        = Slot1Bit(self)
        self.op         = op
        self.in_0_wire  = Wire1Bit(a, self.in_0)

        DesignContext.get_current().add_gate(self)

    @property
    def inputs(self): return [self.in_0]
    @property
    def outputs(self): return [self.out]

    def update(self):
        if self.op == "not":
            self.out.value = (~self.in_0.value) & 1
        else:
            raise ValueError(f"Unknown operation: {self.op}")

class BinaryGate1Bit:
    def __init__(self, a : Slot1Bit, b : Slot1Bit, op):
        self.in_0       = Slot1Bit(self)
        self.in_1       = Slot1Bit(self)
        self.out        = Slot1Bit(self)
        self.op         = op
        self.in_0_wire  = Wire1Bit(a, self.in_0)
        self.in_1_wire  = Wire1Bit(b, self.in_1)

        DesignContext.get_current().add_gate(self)

    @property
    def inputs(self): return [self.in_0, self.in_1]
    @property
    def outputs(self): return [self.out]

    def update(self):
        if self.op == "and":
            self.out.value = self.in_0.value & self.in_1.value
        elif self.op == "or":
            self.out.value = self.in_0.value | self.in_1.value
        elif self.op == "xor":
            self.out.value = self.in_0.value ^ self.in_1.value
        elif self.op == "nand":
            self.out.value = ~(self.in_0.value & self.in_1.value)
        elif self.op == "nor":
            self.out.value = ~(self.in_0.value | self.in_1.value)
        else:
            raise ValueError(f"Unknown operation: {self.op}")
        
class FlipFlop1Bit:
    def __init__(self):
        self.d      = Slot1Bit(self)
        self.q      = Slot1Bit(self)
        self.q_bar  = Slot1Bit(self)
        DesignContext.get_current().add_flip_flop(self)

    def update(self):
        self.q.value        = self.d.value & 1
        self.q_bar.value    = (~self.q.value) & 1

    @property
    def inputs(self): return [self.d]
    @property
    def outputs(self): return [self.q, self.q_bar]

class TwosComplementInverterNBit:
    def __init__(self, width):
        self.width = width
        self.in_0  = [Slot1Bit(self) for _ in range(width)]
        self.out   = [None for _ in range(width)] # we're gonna fill this later

        # The boolean formula for sum is:
        # inv_a    = ~a + 1

        for i in range(width):
            a           = self.in_0[i]
            not_a       = UnaryGate1Bit(a, "not")
            self.out[i] = not_a.out
        
        # now add 1
        carry = None
        for i in range(width):
            a = self.out[i]

            if carry:
                a_xor_carry = BinaryGate1Bit(a, carry.out, "xor")
                sum         = a_xor_carry
                carry       = BinaryGate1Bit(a, carry.out, "and")
            else:
                sum         = BinaryGate1Bit(a, Slot1Bit(self, 1), "xor")
                carry       = BinaryGate1Bit(a, Slot1Bit(self, 1), "and")

            self.out[i]     = sum.out
    
    @property
    def inputs(self): return self.in_0
    @property
    def outputs(self): return self.out

class Multiplier_1_58_Bit:
    def __init__(self):
        self.a    = [Slot1Bit(self) for _ in range(2)]
        self.b    = [Slot1Bit(self) for _ in range(2)]
        self.out  = [None for _ in range(2)] # we're gonna fill this later

        # The boolean formula for sum is:
        # out_0 = a_0 AND b_0
        # out_1 = ((~a_1 AND a_0) AND (b_0 AND b_1)) OR ((a_1 AND a_0) AND (b_0 AND ~b_1))

        a_0 = self.a[0]
        a_1 = self.a[1]
        b_0 = self.b[0]
        b_1 = self.b[1]

        a_0_and_b_0 = BinaryGate1Bit(a_0, b_0, "and")
        self.out[0] = a_0_and_b_0.out

        # This probs could be optimized
        not_a_1                             = UnaryGate1Bit(a_1, "not")
        not_b_1                             = UnaryGate1Bit(b_1, "not")
        a_1_and_a_0                         = BinaryGate1Bit(a_1, a_0, "and")
        b_0_and_b_1                         = BinaryGate1Bit(b_0, b_1, "and")
        not_a_1_and_a_0                     = BinaryGate1Bit(not_a_1.out, a_0, "and")
        not_b_1_and_b_0                     = BinaryGate1Bit(not_b_1.out, b_0, "and")
        not_a_1_and_a_0_and_b_0_and_b_1     = BinaryGate1Bit(not_a_1_and_a_0.out, b_0_and_b_1.out, "and")
        a_1_and_a_0_and_b_0_and_not_b_1     = BinaryGate1Bit(a_1_and_a_0.out, not_b_1_and_b_0.out, "and")
        out_1                               = BinaryGate1Bit(not_a_1_and_a_0_and_b_0_and_b_1.out, a_1_and_a_0_and_b_0_and_not_b_1.out, "or")
        self.out[1] = out_1.out


    @property
    def inputs(self): return [self.a_0, self.a_1, self.b_0, self.b_1]
    @property
    def outputs(self): return [self.out_0, self.out_1]

class AdderNBit:
    """
        N-bit adder
    """
    def __init__(self, width):
        self.width = width
        self.in_0  = [Slot1Bit(self) for _ in range(width)]
        self.in_1  = [Slot1Bit(self) for _ in range(width)]
        self.out   = [None for _ in range(width)] # we're gonna fill this later

        # connect the wires
        # The truth table for carry add is as follows:
        # 3 bits of input, a, b, carry and 2 bits of output, sum, carry'
        # 3 bits is 8 combinations or 2^3
        # | in_0 | in_1 | carry | out | carry' |
        # |------|------|-------|-----|--------|
        # |  0   |  0   |   0   |  0  |   0    |
        # |  0   |  0   |   1   |  1  |   0    |
        # |  0   |  1   |   0   |  1  |   0    |
        # |  0   |  1   |   1   |  0  |   1    |
        # |  1   |  0   |   0   |  1  |   0    |
        # |  1   |  0   |   1   |  0  |   1    |
        # |  1   |  1   |   0   |  0  |   1    |
        # |  1   |  1   |   1   |  1  |   1    |
        # |------|------|-------|-----|--------|
        # The boolean formula for sum is(compressed form):
        # sum    = a XOR b XOR carry
        # carry' = (a AND b) OR (carry AND (a XOR b))

        # build the logic network
        carry_n = None
        for i in range(width):
            a           = self.in_0[i]
            b           = self.in_1[i]
            a_xor_b     = BinaryGate1Bit(a, b, "xor")
            a_and_b     = BinaryGate1Bit(a, b, "and")

            if carry_n:
                a_xor_b_xor_carry   = BinaryGate1Bit(a_xor_b.out, carry_n.out, "xor")
                sum                 = a_xor_b_xor_carry.out
                carry_and_a_xor_b   = BinaryGate1Bit(carry_n.out, a_xor_b.out, "and")
                carry_n             = BinaryGate1Bit(a_and_b.out, carry_and_a_xor_b.out, "or")
            else:
                carry_n = a_and_b
                sum     = a_xor_b.out
            self.out[i] = sum

        # DesignContext.get_current().add_gate(self)
    
    @property
    def inputs(self): return self.in_0 + self.in_1
    @property
    def outputs(self): return self.out

class ValueNBit:
    def __init__(self, value, width):
        max_value = (1 << width) - 1
        self.value = value & max_value

    def __getitem__(self, index):
        return (self.value >> index) & 1

    def __setitem__(self, index, value):
        mask = 1 << index
        if value:
            self.value |= mask
        else:
            self.value &= ~mask

def get_val(slot):
    if isinstance(slot, Slot1Bit): return slot.value
    elif isinstance(slot, FlipFlop1Bit): return slot.q.value
    else: raise ValueError(f"Unknown type {type(slot)}")

def gather_outputs_into_a_number(slots):
    if isinstance(slots, list):
        return sum([get_val(slot) << i for i, slot in enumerate(slots)])
    else:
        return get_val(slots)

def test_add():
    import random
    for i in range(64):
        DesignContext.clean()
        width = 16
        max_value = (1 << width) - 1
        _a = random.randint(0, max_value)
        _b = random.randint(0, max_value)
        a  = ValueNBit(_a, width)
        b  = ValueNBit(_b, width)
        adder = AdderNBit(width)
        for i in range(width):
            adder.in_0[i].value = a[i]
            adder.in_1[i].value = b[i]
        DesignContext.get_current().update()
        result = gather_outputs_into_a_number(adder.out)
        gt = (_a + _b) & max_value
        # print(f"eval {_a} + {_b} = (gt){_a + _b} vs {result}")
        if 0:
            print(f"eval")
            print(f"    {bin(_a)[2:].zfill(width)}")
            print(f"    {bin(_b)[2:].zfill(width)}")
            print(f"gt: {bin(gt)[2:].zfill(width)}")
            print(f"ev: {bin(result)[2:].zfill(width)}")

        assert result == (gt), f"Expected {(gt)}, got {result}"
    

def test_twos_complement_inverter():
    import random
    for i in range(64):
        DesignContext.clean()
        width       = 16
        max_value   = (1 << width) - 1
        _a          = random.randint(0, max_value)
        a           = ValueNBit(_a, width)
        inverter    = TwosComplementInverterNBit(width)
        for i in range(width):
            inverter.in_0[i].value = a[i]
        DesignContext.get_current().update()
        result  = gather_outputs_into_a_number(inverter.out)
        gt      = (-_a) & max_value
        # print(f"eval ~{_a} + 1 = (gt){(~_a + 1)} vs {result}")
        if 0:
            print(f"eval")
            print(f"    {bin(_a)[2:].zfill(width)}")
            print(f"gt: {bin(gt)[2:].zfill(width)}")
            print(f"ev: {bin(result)[2:].zfill(width)}")

        assert result == (gt), f"Expected {(gt)}, got {result}"

def test_multiplier_1_58_bit():
    import random
    for i in range(64):
        DesignContext.clean()
        a = random.randint(-1, 1)
        b = random.randint(-1, 1)
        multiplier = Multiplier_1_58_Bit()
        multiplier.a[0].value = (a >> 0) & 1
        multiplier.a[1].value = (a >> 1) & 1
        multiplier.b[0].value = (b >> 0) & 1
        multiplier.b[1].value = (b >> 1) & 1
        DesignContext.get_current().update()
        result = (multiplier.out[1].value << 1) | multiplier.out[0].value
        gt = (a * b) & 3
        assert result == gt, f"Expected {gt}, {a} * {b} got {result}"

def test_flip_flop():
    DesignContext.clean()
    flip_flop       = FlipFlop1Bit()
    flip_flop.d.value = 1
    assert flip_flop.q.value == 0, f"Expected 0, got {flip_flop.q.value}"

    DesignContext.get_current().update()
    assert flip_flop.q.value == 1, f"Expected 1, got {flip_flop.q.value}"
    
    flip_flop.d.value = 0
    DesignContext.get_current().update()
    assert flip_flop.q.value == 0, f"Expected 0, got {flip_flop.q.value}"

def test_flip_flop_2():
    import random

    for i in range(16):
        DesignContext.clean()
        
        width = 16
        a = [FlipFlop1Bit() for _ in range(16)]
        b = [FlipFlop1Bit() for _ in range(16)]
        c = [FlipFlop1Bit() for _ in range(16)]
        d = [FlipFlop1Bit() for _ in range(16)]
        _a = random.randint(0, (1 << width) - 1)
        _b = random.randint(0, (1 << width) - 1)
        _d = random.randint(0, (1 << width) - 1)

        adder_0 = AdderNBit(width)
        adder_1 = AdderNBit(width)

        for i in range(width):
            a[i].d.value = (_a >> i) & 1
            b[i].d.value = (_b >> i) & 1
            # c[i].d.value = 0

            Wire1Bit(a[i].q,            adder_0.in_0[i])
            Wire1Bit(b[i].q,            adder_0.in_1[i])
            Wire1Bit(adder_0.out[i],    c[i].d)
            Wire1Bit(c[i].q,            adder_1.in_0[i])
            Wire1Bit(d[i].q,            adder_1.in_1[i])

        DesignContext.get_current().update() # trigger inputs, a, b to be loaded into flip-flops
        gt_a_plus_b = (_a + _b) & ((1 << width) - 1)
        assert gather_outputs_into_a_number(adder_0.out) == gt_a_plus_b, f"Expected {gt_a_plus_b}, got {gather_outputs_into_a_number(adder_0.out)}"
        assert gather_outputs_into_a_number(adder_1.out) == 0, f"Expected {0} got {gather_outputs_into_a_number(c)}"


        for i in range(width):
            d[i].d.value = (_d >> i) & 1
        DesignContext.get_current().update() # load d into flip-flops and execute next adder

        assert gather_outputs_into_a_number([i.d for i in c]) == gt_a_plus_b, f"Expected {gt_a_plus_b}, got {gather_outputs_into_a_number([i.d for i in c])}"

        DesignContext.get_current().update()

        gt_c_plus_d = (_d + gt_a_plus_b) & ((1 << width) - 1)
        assert gather_outputs_into_a_number(adder_0.out) == gt_a_plus_b        
        assert gather_outputs_into_a_number(adder_1.out) == gt_c_plus_d, f"Expected {gt_c_plus_d}, got {gather_outputs_into_a_number(adder_1.out)}"
        # print(f"{gather_outputs_into_a_number(adder_1.out)} == {gt_c_plus_d}")

if __name__ == "__main__":

    test_twos_complement_inverter()
    test_add()
    test_multiplier_1_58_bit()
    test_flip_flop()
    test_flip_flop_2()

    print("All tests passed!")