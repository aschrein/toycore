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
    def __init__(self, parent):
        self.value          = 0
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

    def run(self):
        self.q.value        = self.d.value
        self.q_bar.value    = ~self.q.value

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

def gather_outputs_into_a_number(slots):
    return sum([slot.value << i for i, slot in enumerate(slots)])

if __name__ == "__main__":
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
    
    print("All tests passed!")