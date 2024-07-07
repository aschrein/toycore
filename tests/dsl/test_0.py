from typing import Any
from toycore.dsl import *

design_text = f"""

Module(Counter4Bit) {{
    Parameters {{
        u32 width : [default=8];

    }}

    Constants {{
        STATE_ORIGIN    = 0;
        STATE_COUNTING  = 1;

    }}

    Inputs {{
        Wire clock   : [width=1];
        Wire enable  : [width=1];
        Wire input_0 : [width=Parameters.width];
    }}

    Outputs {{
        Wire count   : [width=8];
    }}

    Registers {{
        Register state   : [width=8];
        Register count   : [width=8];
    }}

    Init {{
        Registers.state <- Constants.STATE_ORIGIN;
        Registers.count <- 0;
    }}

    Trigger(clock_trigger) {{
        .condition = (posedge(Inputs.clock) or posedge(Inputs.reset));
    }}
    
    Logic(clock_trigger) {{
        if(enable) {{
            Registers.count <- prev(Registers).count + 1;
        }} else {{
            Registers.count <- 0b0000;
        }}
    }}
}}

"""

modules = parse_modules_from_text(design_text)

for name, trigger in modules[0].triggers.items():
    print(f"Trigger {name} condition: {trigger.condition}")