"""

Here we have utility functions for:
tokenizing, parsing, and interpreting a simple domain-specific language (DSL).

"""

from enum import Enum
from dataclasses import dataclass
from .utils import *
from typing import Any

class TokenType(Enum):
    NUMBER      = 1
    LITERAL     = 2
    STRING      = 3
    OPERATOR    = 4
    SPECIAL     = 5

# console color codes
class ConsoleColor:
    RED     = '\033[91m'
    GREEN   = '\033[92m'
    YELLOW  = '\033[93m'
    BLUE    = '\033[94m'
    PURPLE  = '\033[95m'
    CYAN    = '\033[96m'
    END     = '\033[0m'

@dataclass
class Token:
    type: TokenType
    value: any
    line: int
    col: int


    def is_float(self):
        return self.type == TokenType.NUMBER and '.' in self.value
    
    def get_float(self):
        return float(self.value)

class TokenStream:

    """
        General purpose token stream for tokenizing a string for a C like language.
    """

    def __init__(self, string_or_tokenlist, _file_path=None, _build_ast=None, _lines=None):
        """
            _build_ast is when we want to build an AST from the tokens inside the try/catch block
        """

        if isinstance(string_or_tokenlist, list):
            """
                If a list of tokens is provided, then we can skip tokenizing the string
            """
            self.tokens = string_or_tokenlist
            self.string = None
            self.lines  = _lines
            self.pos    = 0
            self.file_path = None
            self.ast = None
            return
        
        assert isinstance(string_or_tokenlist, str), "Expected string or list of tokens"

        string = string_or_tokenlist

        if _file_path is None:
            """
                Create a temporary file to store the string
            """
            _file_path = get_or_create_tmp() / ("dsl/" + str(hash(string) & 0xff) + ".dsl")
            mkdir_recursive(_file_path.parent)
            with open(_file_path, "w") as f:
                f.write(string)

        self.tokens     = []
        self.string     = string
        self.lines      = string.split('\n')
        self.pos        = 0
        self.file_path  = _file_path if _file_path else "NONE"

        line    = 0
        col     = 0
        cur_col = 0
        cur_str = ""
        idx     = 0
        def _flush_token():
            nonlocal cur_str
            nonlocal line
            nonlocal col
            nonlocal idx
            nonlocal self
            if cur_str != "":
                self.tokens.append(Token(TokenType.LITERAL, cur_str, line, col))
                cur_str = ""
        try:
            while idx < len(string):
                self.pos = len(self.tokens) - 1
                c       = string[idx]
                nc      = string[idx + 1] if idx + 1 < len(string) else None
                nnc     = string[idx + 2] if idx + 2 < len(string) else None
                nnnc    = string[idx + 3] if idx + 3 < len(string) else None
                dop     = c
                trip    = c
                if nc:
                    dop = c + nc
                    if nnc:
                        trip = c + nc + nnc
                if c == '\n':
                    line += 1
                    cur_col = 0
                    _flush_token()
                elif c == ' ' or c == '\t':
                    _flush_token()
                # skip comments
                elif c == '/' and nc == '/':
                    _flush_token()

                    while nc and nc != '\n':
                        idx += 1
                        nc = string[idx + 1] if idx + 1 < len(string) else None
                
                # tripple quoted strings
                elif (c == '"' and nc == '"' and nnc == '"') or (c == "'" and nc == "'" and nnc == "'"):
                    _flush_token()
                    cur_str = ""
                    idx += 2
                    nc = string[idx + 1] if idx + 1 < len(string) else None
                    while nc and nnc and nnnc and (nc != c or nnc != c or nnnc != c):
                        cur_str += nc
                        idx     += 1
                        nc      = string[idx + 1] if idx + 1 < len(string) else None
                        nnc     = string[idx + 2] if idx + 2 < len(string) else None
                        nnnc    = string[idx + 3] if idx + 3 < len(string) else None
                    assert nc == c and nnc == c and nnnc == c, f"Expected closing tripple quote at line {line}, col {col}"
                    idx += 3
                    self.tokens.append(Token(TokenType.STRING, cur_str, line, col))
                    cur_str = ""
                # strings
                elif c == '"' or c == "'":
                    _flush_token()
                    cur_str = ""
                    while nc and nc != c:
                        if nc == '\n':
                            line += 1
                        cur_str += nc
                        idx     += 1
                        nc      = string[idx + 1] if idx + 1 < len(string) else None
                    assert nc == c, f"Expected closing quote at line {line}, col {col}"
                    idx += 1
                    self.tokens.append(Token(TokenType.STRING, cur_str, line, col))
                    cur_str = ""
                elif c.isdigit()                                            \
                        or (c == '.' and nc and nc.isdigit())               \
                        or (c == '+' and nc and nc.isdigit())               \
                        or (c == '-' and nc and nc.isdigit())               \
                        or (c == '0' and nc and (nc == 'x' or nc == 'X'))   \
                        or (c == '0' and nc and (nc == 'b' or nc == 'B'))   \
                                :
                    if len(cur_str) == 0:
                        _flush_token()
                        cur_str += c
                        if nc in ['x', 'X', 'b', 'B']:
                            cur_str += nc
                            idx += 1
                            nc = string[idx + 1] if idx + 1 < len(string) else None

                        is_science = False
                        is_float = False
                        while nc and (nc.isdigit() or nc == '.' or nc == 'e' or nc == 'E' or is_science and (nc == '+' or nc == '-')):
                            if nc == 'e' or nc == 'E':
                                is_science = True
                            elif nc == '.' and not is_science:
                                is_float = True

                            cur_str += nc
                            idx += 1
                            nc = string[idx + 1] if idx + 1 < len(string) else None
                        # print(f"parsed number {cur_str} at line {line}, col {col}")
                        self.tokens.append(Token(TokenType.NUMBER, cur_str.lower(), line, col))
                        cur_str = ""
                    else:
                        
                        """
                            Continue adding to the current token
                        """
                        cur_str += c
                elif (cur_str == "" and dop in ["or", "OR"]) \
                        or (cur_str == "" and trip in ["and", "AND", "xor", "XOR"]) \
                        or c in ['+', '-', '*', '/', '%', '>', '<', '=',
                                 '!', '&', '|', '^', '~', '?', ':', ';',
                                 ',', '.', '@', '#', '$', '`', '\\', '/',
                                 '(', ')', '{', '}', '[', ']']:
                    _flush_token()

                    # check if operator is a double operator
                    if nc and dop in ['==', '!=', '>=', '<=', '&&', '||', '++',
                                   '--', '+=', '-=', '*=', '/=', '%=', '<<',
                                   '>>', '<-', '->', '<=', '=>', '::', "or", "OR"]:
                        self.tokens.append(Token(TokenType.OPERATOR, dop, line, col))
                        idx += 1
                    # check if operator is a tripple operator
                    elif nnc and trip in ['and', 'AND', 'xor', 'XOR']:
                        self.tokens.append(Token(TokenType.OPERATOR, trip, line, col))
                        idx += 2
                    
                    # check if character is a single operator
                    elif c in ['+', '-', '*', '/', '%', '>', '<', '=', '!',
                               '&', '|', '^', '~', '?', ':', ';', ',', '.',
                               '@', '#', '$']:
                        self.tokens.append(Token(TokenType.OPERATOR, c, line, col))
                    else:
                        self.tokens.append(Token(TokenType.SPECIAL, c, line, col))
                
                else:
                    if len(cur_str) == 0:
                        col = cur_col - 1
                    cur_str += c
                cur_col += 1
                idx += 1

            _flush_token()
            self.pos = 0
            self.ast = None
            if _build_ast:
                self.ast = _build_ast(self)
            
        except Exception as e:
            self.print_error_at_current(f"Error while tokenizing: {e}")
            raise e

    def get_line(self, line):
        if self.lines is None:
            return ""
        return self.lines[line]
    
    def get_list_until(self, strings, consume=True):
        """
            return a list of tokens until a string is found
        """
        if not isinstance(strings, list):
            strings = [strings]
        tokens = []
        while True:
            n = self.peek()
            if n is None or n.value in strings:
                if consume:
                    self.move_forward()
                break
            tokens.append(self.next())
        return tokens

    def unwrap_parentheses(self):
        """
            return a list of tokens inside parentheses
        """
        assert self.consume('('), f"Expected '(' but got {self.peek().value}"
        tokens = []
        while not self.consume(')'):
            tokens.append(self.next())
        return tokens

    def print_error_at_current(self, message):
        token = self.tokens[min(self.pos, len(self.tokens) - 1)]
        # put red color on error message
        print(f"{ConsoleColor.RED}", end="")
        print(f"**************************************************")
        print(f"Error at line {token.line}, col {token.col}: {message}")
        print(f"{self.file_path}:{token.line + 1}")
        print(f"{self.get_line(token.line)}")
        # put cursor at col
        print(f"{'-' * token.col}^")
        print(f"**************************************************")
        print(f"{ConsoleColor.END}", end="")

    def get_token_groups_in_between(self, start, end, separator):
        """
            return a list of tokens groups in between start and end
        """
        assert self.consume(start), f"Expected '{start}' but got {self.peek().value}"
        token_groups = []
        current_group = []
        while not self.consume(end):
            if self.consume(separator):
                if len(current_group) > 0:
                    token_groups.append(current_group)
                current_group = []
            else:
                current_group.append(self.next())
        if len(current_group) > 0:
            token_groups.append(current_group)
        return token_groups

    def consume(self, string):
        """
            consume a token if it matches the string
        """
        n = self.peek()
        if n.value == string:
            self.pos += 1
            return True
        return False

    def peek(self):
        """
            return the next token without moving the cursor
        """
        if self.pos < len(self.tokens):
            return self.tokens[self.pos]
        return None

    def next(self):
        """
            return the next token and move the cursor
        """
        token = self.peek()
        if token:
            self.pos += 1
        return token

    def has_more_tokens(self):
        """
            return True if there are more tokens to consume
        """
        return self.pos < len(self.tokens)

    def eof(self):
        """
            return True if we have reached the end of the token stream
        """
        return self.pos >= len(self.tokens)

    def move_back(self):
        """
            move the cursor back by one token
        """
        self.pos -= 1
    
    def move_forward(self):
        """
            move the cursor forward by one token
        """
        self.pos += 1

# Lets get some basic arithmetic expression parsing

class ASTNode:
    def __init__(self, token=None):
        self.token      = token
        self.children   = []

    def add_child(self, child):
        self.children.append(child)

    def __str__(self):
        return f"ASTNode({self.token})"

class ExprType(Enum):
    LITERAL     = 0
    UNARY       = 1
    BINARY      = 2
    PARANTHESIS = 3
    CALL        = 4


@dataclass
class Expr:
    type: ExprType
    token: Token
    args: list

    @property
    def value(self): return self.token.value

    @property
    def left(self): return self.args[0]
    @property
    def right(self): return self.args[1]

    def __repr__(self) -> str:
        if self.type == ExprType.LITERAL:
            return f"{self.value}"
        elif self.type == ExprType.UNARY:
            return f"{self.value} ({self.args[0]})"
        elif self.type == ExprType.BINARY:
            return f"""Binary({self.value})
(({self.args[0]})
({self.args[1]}))"""
        elif self.type == ExprType.PARANTHESIS:
            return f"({self.args[0]})"
        elif self.type == ExprType.CALL:
            return f"{self.value}({self.args})"
        return f"Expr({self.type}, {self.token}, {self.args})"

def recursive_parse_expression(tk: TokenStream):
    """
        Parse an expression from the token stream.
        example:
            (posedge(Inputs.clock) or posedge(Inputs.reset)):
            parenthestis:
                binary(or):
                    args: [
                        0:  call: posedge
                                args: [
                                    0: binary(.):
                                            args: [
                                                0:  literal: Inputs
                                                1:  literal: clock
                                            ]
                                ]
                        1:  call: posedge
                                args: [
                                    0: binary(.):
                                            args: [
                                                0:  literal: Inputs
                                                1:  literal: reset
                                            ]
                                ]

    """
    c = tk.next()
    if not c: return None
    # print (f"parsing token {c}")
    if c.value == '(':
        p = Expr(ExprType.PARANTHESIS, c, [recursive_parse_expression(tk)])
        # print(f"PARANTHESIS {p}")
        assert tk.consume(')'), f"Expected ')' but got {tk.peek().value}"
        return p

    n = tk.peek()
    # print (f"peeking token {n}")
    if n is None:
        return Expr(ExprType.LITERAL, c, [])
    if n.value == '(':
        tk.next()
        p = Expr(ExprType.CALL, c, [recursive_parse_expression(tk)])
        # print(f"CALL {p}")
        assert tk.consume(')'), f"Expected ')' but got {tk.peek().value}"
        return p
    elif n.type == TokenType.OPERATOR:
        tk.next()
        op = Expr(ExprType.BINARY, n, [Expr(ExprType.LITERAL, c, []), recursive_parse_expression(tk)])
        # print (f"binary {op}")
        return op

    else:
        return Expr(ExprType.LITERAL, c, [])

def _recursive_parse_expression(tk: TokenStream):
    c = tk.next()
    if not c: return None
    if c.value == ')':
        tk.move_back()
        return None
    if c.value == '(':
        p = Expr(ExprType.PARANTHESIS, c, [recursive_parse_expression(tk)])
        print(f"PARANTHESIS {p}")
        assert tk.consume(')'), f"Expected ')' but got {tk.peek().value}"
        return p

    # print(f"parsing token {c}")
    if c.type == TokenType.OPERATOR: # unary operator
        return Expr(ExprType.UNARY, c, [recursive_parse_expression(tk)])

    n = tk.next()
    if n is None:
        return Expr(ExprType.LITERAL, c, [])
    elif n.value == ')':
        tk.move_back()
        return Expr(ExprType.LITERAL, c, [])
    elif n.type == TokenType.OPERATOR: # binary operator
        return Expr(ExprType.BINARY, n, [Expr(ExprType.LITERAL, c, []), recursive_parse_expression(tk)])
    elif n.value == '(':
        # tk.move_back()
        p = Expr(ExprType.CALL, c, [recursive_parse_expression(tk)])
        assert tk.consume(')'), f"Expected ')' but got {tk.peek().value}"
        return p
    else:   
        return Expr(ExprType.LITERAL, c, [])
    


if __name__ == "__main__":
    
    string ="""
    // this is a comment
    x = 10;
    y : int = 20;
    t = "hello";
    x = '''-
    this is a tripple quoted string
    ''';

    Function test(x: int, y: int) -> int {
        return x + y;
    }

    test(10, 20);    

    num : f32 = 10.0e-10;
    num : f32 = 10.0e+10;
    num : f32 = 10.0e10;
    num : f32 = 10.0E10;
    num : f32 = f32(1.2);

    z"""
    tk = TokenStream(string, "test.dsl")
    print(tk.tokens)
    tk.print_error_at_current("test error")
    
    for t in tk.tokens:
        if t.is_float():
            print(f"{t.value} is a float = {t.get_float()}")

    e = recursive_parse_expression(TokenStream("1 + 2 * 3", "test.dsl"))

    print(e)
    pass


class Trigger:
    def __init__(self, ts: TokenStream):
        assert ts.consume("Trigger")
        self.name = ts.unwrap_parentheses()[0].value
        self.condition = []
        if ts.consume("{"):
            while not ts.consume("}"):
                if ts.consume("."):
                    if ts.consume("condition"):
                        assert ts.consume("=")
                        l = ts.get_list_until(";")
                        print(f"condition: {l}")
                        self.condition = recursive_parse_expression(TokenStream(l))
                    else:
                        assert False, f"Unexpected token {ts.peek()}"
                else:
                    assert False, f"Unexpected token {ts.peek()}"

    def __repr__(self):
        return f"Trigger({self.name}) {{\n    .condition = {self.condition}\n}}"

def parse_attributes(ts: TokenStream):
    """
        Parses attributes from the token stream into a dictionary.
        example:
            [width=1]
            [storage = device]
    """
    assert ts.consume("[")
    attributes = {}
    while not ts.consume("]"):
        key = ts.next().value
        assert ts.consume("="), f"Expected '=' after key {key}"
        value = ts.get_list_until([",", ";", "]"], consume=False)
        # print(f"key: {key}, value: {value}")
        attributes[key] = value
    return attributes

@dataclass
class Variable:
    template_args: list
    type        : str
    name        : str
    attributes  : dict

    def __repr__(self):
        return f"Variable({self.type} {self.name}, {self.attributes})"

    def get_attribute(self, key: str, default: Any = None):
        return self.attributes.get(key, default)

def parse_variable_definition(ts: TokenStream):
    """
        Parses a variable definition from the token stream.
        example:
            Signal clock                 : [width=1];
            Tensor<f32, 4, 4>    a       : [storage = device];
    """
    type = ts.next().value
    template_args = []
    if ts.consume("<"):
        ts.move_back()
        template_args = ts.get_token_groups_in_between(start="<", end=">", separator=",")
    name = ts.next().value
    assert ts.consume(":")
    attributes = parse_attributes(ts)
    assert ts.consume(";")
    return Variable(template_args, type, name, attributes)

class Module:
    def __init__(self, ts: TokenStream):
        assert ts.consume("Module")
        self.name       = ts.unwrap_parentheses()[0].value
        self.inputs     = {}
        self.outputs    = {}
        self.parameters = {}
        self.registers  = {}
        self.logic      = []
        self.inits      = []
        self.triggers   = {}
        self.submodules = {}
        self.constants  = {}
        
        assert ts.consume("{")
        while ts.has_more_tokens():
            tk  = ts.peek()
            if ts.consume("Inputs"):
                assert ts.consume("{")
                while not ts.consume("}"):
                    var = parse_variable_definition(ts)
                    self.inputs[var.name] = var

            elif ts.consume("Constants"):
                assert ts.consume("{")
                while not ts.consume("}"):
                    name = ts.next().value
                    assert ts.consume("="), f"Expected '=' after constant name {name} but got {ts.peek()}"
                    value = ts.next().value
                    self.constants[name] = value
                    assert ts.consume(";")
            elif ts.consume("Init"):
                assert ts.consume("{")
                while not ts.consume("}"):
                    # TODO
                    ts.move_forward()

            elif ts.consume("Parameters"):
                assert ts.consume("{")
                while not ts.consume("}"):
                    var = parse_variable_definition(ts)
                    self.parameters[var.name] = var

            elif ts.consume("Outputs"):
                assert ts.consume("{")
                while not ts.consume("}"):
                    var = parse_variable_definition(ts)
                    self.outputs[var.name] = var

            elif ts.consume("Registers"):
                assert ts.consume("{")
                while not ts.consume("}"):
                    var = parse_variable_definition(ts)
                    self.registers[var.name] = var

            elif ts.consume("Trigger"):
                ts.move_back()
                trigger = Trigger(ts)
                self.triggers[trigger.name] = trigger

            elif ts.consume("Logic"):
                trigger = ts.unwrap_parentheses()
                assert ts.consume("{")
                while not ts.consume("}"):
                    if ts.consume("if"):
                        condition = ts.unwrap_parentheses()
                        assert ts.consume("{")
                        while not ts.consume("}"):
                            # TODO
                            ts.move_forward()
                        n = ts.next()
                        if n.value == "else":
                            assert ts.consume("{")
                            while not ts.consume("}"):
                                # TODO
                                ts.move_forward()
                    # TODO
                    ts.move_forward()

            elif ts.consume("}"):
                break
            else:
                assert False, f"Unexpected token {tk}"

    def __repr__(self):
        _repr = f"Module({self.name}) {{\n"
        # append inputs
        _repr += f"    Inputs() {{\n"
        for name, inp in self.inputs.items():
            _repr += f"        {inp}\n"
        _repr += f"    }}\n"
        # append outputs
        _repr += f"    Outputs() {{\n"
        for name, out in self.outputs.items():
            _repr += f"        {out}\n"
        _repr += f"    }}\n"
        # append registers
        _repr += f"    Registers() {{\n"
        for name, reg in self.registers.items():
            _repr += f"        {reg}\n"
        _repr += f"    }}\n"

        # append triggers
        for name, trigger in self.triggers.items():
            _repr += f"    Trigger({name}) {{\n"
            _repr += f"        {trigger.condition}\n"
            _repr += f"    }}\n"
        # append logic
        for logic in self.logic:
            _repr += f"    Logic() {{\n"
            _repr += f"        {logic}\n"
            _repr += f"    }}\n"
        _repr += "}"
        return _repr

def parse_modules_from_text(design_text:str):
    modules = []

    def parse_module(ts: TokenStream):
        while ts.has_more_tokens():
            tk = ts.peek()
            if tk.value == "Module":
                m = Module(ts)
                modules.append(m)
            else:
                assert False, f"Unexpected token {tk}"
            
    ts = TokenStream(design_text, _build_ast=parse_module)

    return modules
