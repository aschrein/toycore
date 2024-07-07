module counter_4bit(
    input  wire [0:0] clk,
    input  wire [0:0] reset,
    input  wire [0:0] enable,
    input  wire [7:0] in_wire_instruction,
    input  wire [7:0] in_wire_input_0,
    input  wire [7:0] in_wire_input_1,
    output wire [7:0] out_wire_result,
    output wire [0:0] out_wire_result_ready,
    output wire [0:0] out_wire_busy,
);

reg [7:0] tmp_reg;

reg [1:0] state;
reg [1:0] next_state;
reg [0:0] reg_busy;

localparam [1:0] 
        STATE_IDLE        = 2'b00,
        STATE_BUSY        = 2'b01;

localparam [7:0] 
        INSTRUCTION_ADD        = 8'b00000000,
        INSTRUCTION_SUB        = 8'b00000001,
        INSTRUCTION_OR         = 8'b00000010,
        INSTRUCTION_AND        = 8'b00000011;

always @(posedge clk or posedge reset) begin
    if (reset) begin
        tmp_reg[7:0] <= 4'b00000000;
        reg_busy[0:0] <= 1'b0;
    end else if (enable) begin
        if (state == STATE_IDLE) begin
            next_state <= STATE_BUSY;
            if (in_wire_instruction == INSTRUCTION_ADD) begin
                tmp_reg[7:0] <= in_wire_input_0[7:0] + in_wire_input_1[7:0];
            end else if (in_wire_instruction == INSTRUCTION_SUB) begin
                tmp_reg[7:0] <= in_wire_input_0[7:0] - in_wire_input_1[7:0];
            end else if (in_wire_instruction == INSTRUCTION_OR) begin
                tmp_reg[7:0] <= in_wire_input_0[7:0] | in_wire_input_1[7:0];
            end else if (in_wire_instruction == INSTRUCTION_AND) begin
                tmp_reg[7:0] <= in_wire_input_0[7:0] & in_wire_input_1[7:0];
            end

            reg_busy[0:0] <= 1'b1;
        end else if (state == STATE_BUSY) begin
            next_state <= STATE_IDLE;
            reg_busy[0:0] <= 1'b0;
        end
        reg_count[3:0] <= reg_count[3:0] + 1;
        state <= next_state;

    end


end

assign out_wire_busy            = reg_busy;
assign out_wire_result          = tmp_reg;


endmodule