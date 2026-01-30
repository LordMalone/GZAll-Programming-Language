#!/usr/bin/env python3
"""
GZAll - Gen Z and Alpha Programming Language Interpreter
A Programming Language Made of Gen Z and Alpha Slang
With User Interface

Created by the President Honourable of the Gen Z and A Population
All rights reserved to the sweed
"""

import tkinter as tk
from tkinter import scrolledtext, messagebox, ttk
import re
from enum import Enum, auto
from dataclasses import dataclass
from typing import Any, List, Dict, Optional, Union
import sys
import io


# ============================================================================
# TOKEN DEFINITIONS
# ============================================================================

class TokenType(Enum):
    """Enumeration of all token types in GZAll language"""
    # Literals
    NUMBER = auto()
    STRING = auto()
    IDENTIFIER = auto()
    
    # Keywords (Gen Z Slang)
    NOCAP = auto()          # print/output
    FR = auto()             # variable declaration
    BUSSIN = auto()         # function definition
    SLAY = auto()           # return
    VIBE = auto()           # if
    MIDCHECK = auto()       # else
    BET = auto()            # while loop
    DEADASS = auto()        # true
    CAP = auto()            # false
    LOWKEY = auto()         # for loop
    HIGHKEY = auto()        # and
    SUS = auto()            # not
    SHEESH = auto()         # comment
    YEET = auto()           # delete
    FINNA = auto()          # will/going to
    GHOST = auto()          # break
    FLEX = auto()           # display
    BRUH = auto()           # else if
    SALTY = auto()          # or
    GOAT = auto()           # max
    SIMP = auto()           # min
    CHAD = auto()           # abs
    DRIP = auto()           # style/format
    RIZZ = auto()           # input
    PERIODT = auto()        # end statement
    SLAPS = auto()          # equals/assign
    RATIO = auto()          # divide
    HITS = auto()           # multiply
    MOOD = auto()           # same as
    AINT = auto()           # not equal
    BIGGER = auto()         # greater than
    SMALLER = auto()        # less than
    VIBECHECK = auto()      # check condition
    BASED = auto()          # valid/correct
    CRINGE = auto()         # invalid/error
    CAUGHT = auto()         # caught in 4k
    
    # Operators
    PLUS = auto()
    MINUS = auto()
    MULTIPLY = auto()
    DIVIDE = auto()
    MODULO = auto()
    ASSIGN = auto()
    EQUAL = auto()
    NOT_EQUAL = auto()
    LESS_THAN = auto()
    GREATER_THAN = auto()
    LESS_EQUAL = auto()
    GREATER_EQUAL = auto()
    
    # Delimiters
    LPAREN = auto()
    RPAREN = auto()
    LBRACE = auto()
    RBRACE = auto()
    LBRACKET = auto()
    RBRACKET = auto()
    COMMA = auto()
    COLON = auto()
    SEMICOLON = auto()
    ARROW = auto()
    
    # Special
    NEWLINE = auto()
    EOF = auto()
    

@dataclass
class Token:
    """Represents a single token in the source code"""
    type: TokenType
    value: Any
    line: int
    column: int
    
    def __repr__(self):
        return f"Token({self.type}, {self.value}, {self.line}:{self.column})"


# ============================================================================
# LEXER (TOKENIZER)
# ============================================================================

class Lexer:
    """Tokenizes GZAll source code into tokens"""
    
    KEYWORDS = {
        'nocap': TokenType.NOCAP,
        'fr': TokenType.FR,
        'bussin': TokenType.BUSSIN,
        'slay': TokenType.SLAY,
        'vibe': TokenType.VIBE,
        'midcheck': TokenType.MIDCHECK,
        'bet': TokenType.BET,
        'deadass': TokenType.DEADASS,
        'cap': TokenType.CAP,
        'lowkey': TokenType.LOWKEY,
        'highkey': TokenType.HIGHKEY,
        'sus': TokenType.SUS,
        'sheesh': TokenType.SHEESH,
        'yeet': TokenType.YEET,
        'finna': TokenType.FINNA,
        'ghost': TokenType.GHOST,
        'flex': TokenType.FLEX,
        'bruh': TokenType.BRUH,
        'salty': TokenType.SALTY,
        'goat': TokenType.GOAT,
        'simp': TokenType.SIMP,
        'chad': TokenType.CHAD,
        'drip': TokenType.DRIP,
        'rizz': TokenType.RIZZ,
        'periodt': TokenType.PERIODT,
        'slaps': TokenType.SLAPS,
        'ratio': TokenType.RATIO,
        'hits': TokenType.HITS,
        'mood': TokenType.MOOD,
        'aint': TokenType.AINT,
        'bigger': TokenType.BIGGER,
        'smaller': TokenType.SMALLER,
        'vibecheck': TokenType.VIBECHECK,
        'based': TokenType.BASED,
        'cringe': TokenType.CRINGE,
        'caught': TokenType.CAUGHT,
    }
    
    def __init__(self, source_code: str):
        self.source = source_code
        self.pos = 0
        self.line = 1
        self.column = 1
        self.tokens = []
        
    def error(self, message: str):
        raise SyntaxError(f"Lexer Error at line {self.line}, column {self.column}: {message}")
        
    def peek(self, offset: int = 0) -> Optional[str]:
        """Look ahead at character without consuming it"""
        pos = self.pos + offset
        if pos < len(self.source):
            return self.source[pos]
        return None
        
    def advance(self) -> Optional[str]:
        """Consume and return current character"""
        if self.pos >= len(self.source):
            return None
        char = self.source[self.pos]
        self.pos += 1
        if char == '\n':
            self.line += 1
            self.column = 1
        else:
            self.column += 1
        return char
        
    def skip_whitespace(self):
        """Skip whitespace characters except newlines"""
        while self.peek() and self.peek() in ' \t\r':
            self.advance()
            
    def skip_comment(self):
        """Skip single-line comments starting with sheesh"""
        while self.peek() and self.peek() != '\n':
            self.advance()
            
    def read_number(self) -> Token:
        """Read a numeric literal"""
        start_line = self.line
        start_col = self.column
        num_str = ''
        has_dot = False
        
        while self.peek() and (self.peek().isdigit() or self.peek() == '.'):
            if self.peek() == '.':
                if has_dot:
                    self.error("Invalid number format")
                has_dot = True
            num_str += self.advance()
            
        value = float(num_str) if has_dot else int(num_str)
        return Token(TokenType.NUMBER, value, start_line, start_col)
        
    def read_string(self) -> Token:
        """Read a string literal"""
        start_line = self.line
        start_col = self.column
        quote_char = self.advance()  # consume opening quote
        string_val = ''
        
        while self.peek() and self.peek() != quote_char:
            if self.peek() == '\\':
                self.advance()
                next_char = self.advance()
                if next_char == 'n':
                    string_val += '\n'
                elif next_char == 't':
                    string_val += '\t'
                elif next_char == '\\':
                    string_val += '\\'
                elif next_char == quote_char:
                    string_val += quote_char
                else:
                    string_val += next_char
            else:
                string_val += self.advance()
                
        if not self.peek():
            self.error("Unterminated string")
            
        self.advance()  # consume closing quote
        return Token(TokenType.STRING, string_val, start_line, start_col)
        
    def read_identifier(self) -> Token:
        """Read an identifier or keyword"""
        start_line = self.line
        start_col = self.column
        ident = ''
        
        while self.peek() and (self.peek().isalnum() or self.peek() == '_'):
            ident += self.advance()
            
        token_type = self.KEYWORDS.get(ident.lower(), TokenType.IDENTIFIER)
        value = ident if token_type == TokenType.IDENTIFIER else ident.lower()
        
        return Token(token_type, value, start_line, start_col)
        
    def tokenize(self) -> List[Token]:
        """Convert source code into list of tokens"""
        while self.pos < len(self.source):
            self.skip_whitespace()
            
            if not self.peek():
                break
                
            char = self.peek()
            start_line = self.line
            start_col = self.column
            
            # Newline
            if char == '\n':
                self.advance()
                self.tokens.append(Token(TokenType.NEWLINE, '\n', start_line, start_col))
                continue
                
            # Numbers
            if char.isdigit():
                self.tokens.append(self.read_number())
                continue
                
            # Strings
            if char in '"\'':
                self.tokens.append(self.read_string())
                continue
                
            # Identifiers and keywords
            if char.isalpha() or char == '_':
                token = self.read_identifier()
                # Handle sheesh comments
                if token.type == TokenType.SHEESH:
                    self.skip_comment()
                    continue
                self.tokens.append(token)
                continue
                
            # Two-character operators
            if char == '=' and self.peek(1) == '=':
                self.advance()
                self.advance()
                self.tokens.append(Token(TokenType.EQUAL, '==', start_line, start_col))
                continue
                
            if char == '!' and self.peek(1) == '=':
                self.advance()
                self.advance()
                self.tokens.append(Token(TokenType.NOT_EQUAL, '!=', start_line, start_col))
                continue
                
            if char == '<' and self.peek(1) == '=':
                self.advance()
                self.advance()
                self.tokens.append(Token(TokenType.LESS_EQUAL, '<=', start_line, start_col))
                continue
                
            if char == '>' and self.peek(1) == '=':
                self.advance()
                self.advance()
                self.tokens.append(Token(TokenType.GREATER_EQUAL, '>=', start_line, start_col))
                continue
                
            if char == '-' and self.peek(1) == '>':
                self.advance()
                self.advance()
                self.tokens.append(Token(TokenType.ARROW, '->', start_line, start_col))
                continue
                
            # Single-character tokens
            single_char_tokens = {
                '+': TokenType.PLUS,
                '-': TokenType.MINUS,
                '*': TokenType.MULTIPLY,
                '/': TokenType.DIVIDE,
                '%': TokenType.MODULO,
                '=': TokenType.ASSIGN,
                '<': TokenType.LESS_THAN,
                '>': TokenType.GREATER_THAN,
                '(': TokenType.LPAREN,
                ')': TokenType.RPAREN,
                '{': TokenType.LBRACE,
                '}': TokenType.RBRACE,
                '[': TokenType.LBRACKET,
                ']': TokenType.RBRACKET,
                ',': TokenType.COMMA,
                ':': TokenType.COLON,
                ';': TokenType.SEMICOLON,
            }
            
            if char in single_char_tokens:
                self.advance()
                self.tokens.append(Token(single_char_tokens[char], char, start_line, start_col))
                continue
                
            self.error(f"Unexpected character: '{char}'")
            
        self.tokens.append(Token(TokenType.EOF, None, self.line, self.column))
        return self.tokens


# ============================================================================
# ABSTRACT SYNTAX TREE (AST) NODES
# ============================================================================

class ASTNode:
    """Base class for all AST nodes"""
    pass


class NumberNode(ASTNode):
    def __init__(self, value):
        self.value = value
        

class StringNode(ASTNode):
    def __init__(self, value):
        self.value = value
        

class BooleanNode(ASTNode):
    def __init__(self, value):
        self.value = value
        

class IdentifierNode(ASTNode):
    def __init__(self, name):
        self.name = name
        

class BinaryOpNode(ASTNode):
    def __init__(self, left, operator, right):
        self.left = left
        self.operator = operator
        self.right = right
        

class UnaryOpNode(ASTNode):
    def __init__(self, operator, operand):
        self.operator = operator
        self.operand = operand
        

class AssignNode(ASTNode):
    def __init__(self, name, value):
        self.name = name
        self.value = value
        

class PrintNode(ASTNode):
    def __init__(self, expressions):
        self.expressions = expressions
        

class InputNode(ASTNode):
    def __init__(self, prompt=None):
        self.prompt = prompt
        

class IfNode(ASTNode):
    def __init__(self, condition, if_body, elif_parts=None, else_body=None):
        self.condition = condition
        self.if_body = if_body
        self.elif_parts = elif_parts or []
        self.else_body = else_body
        

class WhileNode(ASTNode):
    def __init__(self, condition, body):
        self.condition = condition
        self.body = body
        

class ForNode(ASTNode):
    def __init__(self, var_name, iterable, body):
        self.var_name = var_name
        self.iterable = iterable
        self.body = body
        

class FunctionDefNode(ASTNode):
    def __init__(self, name, params, body):
        self.name = name
        self.params = params
        self.body = body
        

class FunctionCallNode(ASTNode):
    def __init__(self, name, args):
        self.name = name
        self.args = args
        

class ReturnNode(ASTNode):
    def __init__(self, value):
        self.value = value
        

class BreakNode(ASTNode):
    pass
    

class ListNode(ASTNode):
    def __init__(self, elements):
        self.elements = elements
        

class IndexNode(ASTNode):
    def __init__(self, object, index):
        self.object = object
        self.index = index
        

class BlockNode(ASTNode):
    def __init__(self, statements):
        self.statements = statements


# ============================================================================
# PARSER
# ============================================================================

class Parser:
    """Parses tokens into an Abstract Syntax Tree"""
    
    def __init__(self, tokens: List[Token]):
        self.tokens = tokens
        self.pos = 0
        self.current_token = self.tokens[0] if tokens else None
        
    def error(self, message: str):
        if self.current_token:
            raise SyntaxError(f"Parser Error at line {self.current_token.line}: {message}")
        raise SyntaxError(f"Parser Error: {message}")
        
    def advance(self):
        """Move to next token"""
        self.pos += 1
        if self.pos < len(self.tokens):
            self.current_token = self.tokens[self.pos]
        else:
            self.current_token = None
            
    def skip_newlines(self):
        """Skip any newline tokens"""
        while self.current_token and self.current_token.type == TokenType.NEWLINE:
            self.advance()
            
    def expect(self, token_type: TokenType):
        """Consume token of expected type or raise error"""
        if not self.current_token or self.current_token.type != token_type:
            self.error(f"Expected {token_type}, got {self.current_token.type if self.current_token else 'EOF'}")
        value = self.current_token.value
        self.advance()
        return value
        
    def parse(self) -> BlockNode:
        """Parse entire program"""
        statements = []
        self.skip_newlines()
        
        while self.current_token and self.current_token.type != TokenType.EOF:
            stmt = self.parse_statement()
            if stmt:
                statements.append(stmt)
            self.skip_newlines()
            
        return BlockNode(statements)
        
    def parse_statement(self):
        """Parse a single statement"""
        self.skip_newlines()
        
        if not self.current_token or self.current_token.type == TokenType.EOF:
            return None
            
        token_type = self.current_token.type
        
        # Variable declaration: fr variableName slaps value
        if token_type == TokenType.FR:
            return self.parse_variable_declaration()
            
        # Print statement: nocap expression
        if token_type == TokenType.NOCAP:
            return self.parse_print()
            
        # If statement: vibe condition
        if token_type == TokenType.VIBE:
            return self.parse_if()
            
        # While loop: bet condition
        if token_type == TokenType.BET:
            return self.parse_while()
            
        # For loop: lowkey variable in iterable
        if token_type == TokenType.LOWKEY:
            return self.parse_for()
            
        # Function definition: bussin functionName
        if token_type == TokenType.BUSSIN:
            return self.parse_function_def()
            
        # Return statement: slay value
        if token_type == TokenType.SLAY:
            return self.parse_return()
            
        # Break statement: ghost
        if token_type == TokenType.GHOST:
            self.advance()
            return BreakNode()
            
        # Assignment or expression statement
        return self.parse_assignment_or_expression()
        
    def parse_variable_declaration(self):
        """Parse: fr varName slaps value"""
        self.expect(TokenType.FR)
        var_name = self.expect(TokenType.IDENTIFIER)
        self.expect(TokenType.SLAPS)
        value = self.parse_expression()
        return AssignNode(var_name, value)
        
    def parse_print(self):
        """Parse: nocap expression1, expression2, ..."""
        self.expect(TokenType.NOCAP)
        expressions = []
        
        if self.current_token and self.current_token.type not in (TokenType.NEWLINE, TokenType.EOF):
            expressions.append(self.parse_expression())
            
            while self.current_token and self.current_token.type == TokenType.COMMA:
                self.advance()
                expressions.append(self.parse_expression())
                
        return PrintNode(expressions)
        
    def parse_if(self):
        """Parse: vibe condition: body [bruh condition: body]* [midcheck: body]"""
        self.expect(TokenType.VIBE)
        condition = self.parse_expression()
        self.expect(TokenType.COLON)
        self.skip_newlines()
        if_body = self.parse_block()
        
        elif_parts = []
        else_body = None
        
        # Handle bruh (elif)
        while self.current_token and self.current_token.type == TokenType.BRUH:
            self.advance()
            elif_condition = self.parse_expression()
            self.expect(TokenType.COLON)
            self.skip_newlines()
            elif_body = self.parse_block()
            elif_parts.append((elif_condition, elif_body))
            
        # Handle midcheck (else)
        if self.current_token and self.current_token.type == TokenType.MIDCHECK:
            self.advance()
            self.expect(TokenType.COLON)
            self.skip_newlines()
            else_body = self.parse_block()
            
        return IfNode(condition, if_body, elif_parts, else_body)
        
    def parse_while(self):
        """Parse: bet condition: body"""
        self.expect(TokenType.BET)
        condition = self.parse_expression()
        self.expect(TokenType.COLON)
        self.skip_newlines()
        body = self.parse_block()
        return WhileNode(condition, body)
        
    def parse_for(self):
        """Parse: lowkey var in iterable: body"""
        self.expect(TokenType.LOWKEY)
        var_name = self.expect(TokenType.IDENTIFIER)
        
        # Expect 'in' keyword (we'll use 'mood' for this)
        if self.current_token and self.current_token.type == TokenType.IDENTIFIER and self.current_token.value == 'in':
            self.advance()
        else:
            self.error("Expected 'in' after loop variable")
            
        iterable = self.parse_expression()
        self.expect(TokenType.COLON)
        self.skip_newlines()
        body = self.parse_block()
        return ForNode(var_name, iterable, body)
        
    def parse_function_def(self):
        """Parse: bussin functionName(params): body"""
        self.expect(TokenType.BUSSIN)
        func_name = self.expect(TokenType.IDENTIFIER)
        self.expect(TokenType.LPAREN)
        
        params = []
        if self.current_token and self.current_token.type != TokenType.RPAREN:
            params.append(self.expect(TokenType.IDENTIFIER))
            while self.current_token and self.current_token.type == TokenType.COMMA:
                self.advance()
                params.append(self.expect(TokenType.IDENTIFIER))
                
        self.expect(TokenType.RPAREN)
        self.expect(TokenType.COLON)
        self.skip_newlines()
        body = self.parse_block()
        return FunctionDefNode(func_name, params, body)
        
    def parse_return(self):
        """Parse: slay value"""
        self.expect(TokenType.SLAY)
        value = None
        if self.current_token and self.current_token.type not in (TokenType.NEWLINE, TokenType.EOF):
            value = self.parse_expression()
        return ReturnNode(value)
        
    def parse_block(self):
        """Parse a block of statements (indentation-based)"""
        statements = []
        
        # Simple block parsing - statements until we hit certain keywords or EOF
        while self.current_token and self.current_token.type not in (
            TokenType.EOF, TokenType.MIDCHECK, TokenType.BRUH
        ):
            # Check if we're at a dedent (new statement at same or lower level)
            if self.current_token.type in (
                TokenType.VIBE, TokenType.BET, TokenType.LOWKEY, 
                TokenType.BUSSIN, TokenType.FR
            ) and statements:
                break
                
            stmt = self.parse_statement()
            if stmt:
                statements.append(stmt)
            self.skip_newlines()
            
            # Simple heuristic: if we get another major keyword, break
            if self.current_token and self.current_token.type in (
                TokenType.MIDCHECK, TokenType.BRUH
            ):
                break
                
        return BlockNode(statements)
        
    def parse_assignment_or_expression(self):
        """Parse assignment or expression statement"""
        expr = self.parse_expression()
        
        # Check for assignment
        if self.current_token and self.current_token.type in (TokenType.ASSIGN, TokenType.SLAPS):
            if not isinstance(expr, IdentifierNode):
                self.error("Invalid assignment target")
            self.advance()
            value = self.parse_expression()
            return AssignNode(expr.name, value)
            
        return expr
        
    def parse_expression(self):
        """Parse expression with operator precedence"""
        return self.parse_or()
        
    def parse_or(self):
        """Parse logical OR (salty)"""
        left = self.parse_and()
        
        while self.current_token and self.current_token.type == TokenType.SALTY:
            op = self.current_token.type
            self.advance()
            right = self.parse_and()
            left = BinaryOpNode(left, op, right)
            
        return left
        
    def parse_and(self):
        """Parse logical AND (highkey)"""
        left = self.parse_comparison()
        
        while self.current_token and self.current_token.type == TokenType.HIGHKEY:
            op = self.current_token.type
            self.advance()
            right = self.parse_comparison()
            left = BinaryOpNode(left, op, right)
            
        return left
        
    def parse_comparison(self):
        """Parse comparison operators"""
        left = self.parse_additive()
        
        while self.current_token and self.current_token.type in (
            TokenType.EQUAL, TokenType.NOT_EQUAL, TokenType.LESS_THAN,
            TokenType.GREATER_THAN, TokenType.LESS_EQUAL, TokenType.GREATER_EQUAL,
            TokenType.MOOD, TokenType.AINT, TokenType.BIGGER, TokenType.SMALLER
        ):
            op = self.current_token.type
            self.advance()
            right = self.parse_additive()
            left = BinaryOpNode(left, op, right)
            
        return left
        
    def parse_additive(self):
        """Parse addition and subtraction"""
        left = self.parse_multiplicative()
        
        while self.current_token and self.current_token.type in (TokenType.PLUS, TokenType.MINUS):
            op = self.current_token.type
            self.advance()
            right = self.parse_multiplicative()
            left = BinaryOpNode(left, op, right)
            
        return left
        
    def parse_multiplicative(self):
        """Parse multiplication, division, and modulo"""
        left = self.parse_unary()
        
        while self.current_token and self.current_token.type in (
            TokenType.MULTIPLY, TokenType.DIVIDE, TokenType.MODULO,
            TokenType.HITS, TokenType.RATIO
        ):
            op = self.current_token.type
            self.advance()
            right = self.parse_unary()
            left = BinaryOpNode(left, op, right)
            
        return left
        
    def parse_unary(self):
        """Parse unary operators"""
        if self.current_token and self.current_token.type in (TokenType.MINUS, TokenType.SUS):
            op = self.current_token.type
            self.advance()
            operand = self.parse_unary()
            return UnaryOpNode(op, operand)
            
        return self.parse_postfix()
        
    def parse_postfix(self):
        """Parse postfix expressions (function calls, indexing)"""
        expr = self.parse_primary()
        
        while True:
            if self.current_token and self.current_token.type == TokenType.LPAREN:
                # Function call
                self.advance()
                args = []
                if self.current_token and self.current_token.type != TokenType.RPAREN:
                    args.append(self.parse_expression())
                    while self.current_token and self.current_token.type == TokenType.COMMA:
                        self.advance()
                        args.append(self.parse_expression())
                self.expect(TokenType.RPAREN)
                expr = FunctionCallNode(expr.name if isinstance(expr, IdentifierNode) else None, args)
                
            elif self.current_token and self.current_token.type == TokenType.LBRACKET:
                # Indexing
                self.advance()
                index = self.parse_expression()
                self.expect(TokenType.RBRACKET)
                expr = IndexNode(expr, index)
                
            else:
                break
                
        return expr
        
    def parse_primary(self):
        """Parse primary expressions"""
        if not self.current_token:
            self.error("Unexpected end of input")
            
        token_type = self.current_token.type
        
        # Numbers
        if token_type == TokenType.NUMBER:
            value = self.current_token.value
            self.advance()
            return NumberNode(value)
            
        # Strings
        if token_type == TokenType.STRING:
            value = self.current_token.value
            self.advance()
            return StringNode(value)
            
        # Booleans
        if token_type == TokenType.DEADASS:
            self.advance()
            return BooleanNode(True)
            
        if token_type == TokenType.CAP:
            self.advance()
            return BooleanNode(False)
            
        # Input: rizz "prompt"
        if token_type == TokenType.RIZZ:
            self.advance()
            prompt = None
            if self.current_token and self.current_token.type == TokenType.LPAREN:
                self.advance()
                if self.current_token.type != TokenType.RPAREN:
                    prompt = self.parse_expression()
                self.expect(TokenType.RPAREN)
            return InputNode(prompt)
            
        # Built-in functions
        if token_type in (TokenType.GOAT, TokenType.SIMP, TokenType.CHAD):
            func_type = token_type
            self.advance()
            self.expect(TokenType.LPAREN)
            args = [self.parse_expression()]
            while self.current_token and self.current_token.type == TokenType.COMMA:
                self.advance()
                args.append(self.parse_expression())
            self.expect(TokenType.RPAREN)
            func_name = {
                TokenType.GOAT: 'goat',
                TokenType.SIMP: 'simp',
                TokenType.CHAD: 'chad'
            }[func_type]
            return FunctionCallNode(func_name, args)
            
        # Identifiers
        if token_type == TokenType.IDENTIFIER:
            name = self.current_token.value
            self.advance()
            return IdentifierNode(name)
            
        # Parenthesized expressions
        if token_type == TokenType.LPAREN:
            self.advance()
            expr = self.parse_expression()
            self.expect(TokenType.RPAREN)
            return expr
            
        # Lists
        if token_type == TokenType.LBRACKET:
            self.advance()
            elements = []
            if self.current_token and self.current_token.type != TokenType.RBRACKET:
                elements.append(self.parse_expression())
                while self.current_token and self.current_token.type == TokenType.COMMA:
                    self.advance()
                    elements.append(self.parse_expression())
            self.expect(TokenType.RBRACKET)
            return ListNode(elements)
            
        self.error(f"Unexpected token: {token_type}")


# ============================================================================
# INTERPRETER / EVALUATOR
# ============================================================================

class BreakException(Exception):
    """Exception to handle break statements"""
    pass


class ReturnException(Exception):
    """Exception to handle return statements"""
    def __init__(self, value):
        self.value = value


class Environment:
    """Symbol table for variable and function storage"""
    def __init__(self, parent=None):
        self.parent = parent
        self.symbols = {}
        
    def define(self, name: str, value: Any):
        self.symbols[name] = value
        
    def get(self, name: str) -> Any:
        if name in self.symbols:
            return self.symbols[name]
        if self.parent:
            return self.parent.get(name)
        raise NameError(f"Undefined variable: {name}")
        
    def set(self, name: str, value: Any):
        if name in self.symbols:
            self.symbols[name] = value
        elif self.parent:
            self.parent.set(name, value)
        else:
            self.symbols[name] = value


class Function:
    """Represents a user-defined function"""
    def __init__(self, name, params, body, closure):
        self.name = name
        self.params = params
        self.body = body
        self.closure = closure


class Interpreter:
    """Interprets and executes the AST"""
    
    def __init__(self, output_callback=None, input_callback=None):
        self.global_env = Environment()
        self.output_callback = output_callback or print
        self.input_callback = input_callback or input
        
    def interpret(self, ast: ASTNode, env: Environment = None):
        """Execute the AST"""
        if env is None:
            env = self.global_env
        return self.evaluate(ast, env)
        
    def evaluate(self, node: ASTNode, env: Environment) -> Any:
        """Evaluate an AST node"""
        if isinstance(node, NumberNode):
            return node.value
            
        if isinstance(node, StringNode):
            return node.value
            
        if isinstance(node, BooleanNode):
            return node.value
            
        if isinstance(node, IdentifierNode):
            return env.get(node.name)
            
        if isinstance(node, BinaryOpNode):
            return self.evaluate_binary_op(node, env)
            
        if isinstance(node, UnaryOpNode):
            return self.evaluate_unary_op(node, env)
            
        if isinstance(node, AssignNode):
            value = self.evaluate(node.value, env)
            env.set(node.name, value)
            return value
            
        if isinstance(node, PrintNode):
            values = [self.evaluate(expr, env) for expr in node.expressions]
            output = ' '.join(str(v) for v in values)
            self.output_callback(output)
            return None
            
        if isinstance(node, InputNode):
            prompt = ''
            if node.prompt:
                prompt = str(self.evaluate(node.prompt, env))
            return self.input_callback(prompt)
            
        if isinstance(node, IfNode):
            condition = self.evaluate(node.condition, env)
            if self.is_truthy(condition):
                return self.evaluate(node.if_body, env)
            for elif_condition, elif_body in node.elif_parts:
                if self.is_truthy(self.evaluate(elif_condition, env)):
                    return self.evaluate(elif_body, env)
            if node.else_body:
                return self.evaluate(node.else_body, env)
            return None
            
        if isinstance(node, WhileNode):
            while self.is_truthy(self.evaluate(node.condition, env)):
                try:
                    self.evaluate(node.body, env)
                except BreakException:
                    break
            return None
            
        if isinstance(node, ForNode):
            iterable = self.evaluate(node.iterable, env)
            for item in iterable:
                env.define(node.var_name, item)
                try:
                    self.evaluate(node.body, env)
                except BreakException:
                    break
            return None
            
        if isinstance(node, FunctionDefNode):
            func = Function(node.name, node.params, node.body, env)
            env.define(node.name, func)
            return None
            
        if isinstance(node, FunctionCallNode):
            return self.call_function(node, env)
            
        if isinstance(node, ReturnNode):
            value = None
            if node.value:
                value = self.evaluate(node.value, env)
            raise ReturnException(value)
            
        if isinstance(node, BreakNode):
            raise BreakException()
            
        if isinstance(node, ListNode):
            return [self.evaluate(elem, env) for elem in node.elements]
            
        if isinstance(node, IndexNode):
            obj = self.evaluate(node.object, env)
            index = self.evaluate(node.index, env)
            return obj[int(index)]
            
        if isinstance(node, BlockNode):
            result = None
            for stmt in node.statements:
                result = self.evaluate(stmt, env)
            return result
            
        raise RuntimeError(f"Unknown node type: {type(node).__name__}")
        
    def evaluate_binary_op(self, node: BinaryOpNode, env: Environment) -> Any:
        """Evaluate binary operations"""
        left = self.evaluate(node.left, env)
        right = self.evaluate(node.right, env)
        op = node.operator
        
        # Arithmetic
        if op == TokenType.PLUS:
            return left + right
        if op == TokenType.MINUS:
            return left - right
        if op in (TokenType.MULTIPLY, TokenType.HITS):
            return left * right
        if op in (TokenType.DIVIDE, TokenType.RATIO):
            if right == 0:
                raise ZeroDivisionError("Division by zero")
            return left / right
        if op == TokenType.MODULO:
            return left % right
            
        # Comparison
        if op in (TokenType.EQUAL, TokenType.MOOD):
            return left == right
        if op in (TokenType.NOT_EQUAL, TokenType.AINT):
            return left != right
        if op in (TokenType.LESS_THAN, TokenType.SMALLER):
            return left < right
        if op in (TokenType.GREATER_THAN, TokenType.BIGGER):
            return left > right
        if op == TokenType.LESS_EQUAL:
            return left <= right
        if op == TokenType.GREATER_EQUAL:
            return left >= right
            
        # Logical
        if op == TokenType.HIGHKEY:
            return self.is_truthy(left) and self.is_truthy(right)
        if op == TokenType.SALTY:
            return self.is_truthy(left) or self.is_truthy(right)
            
        raise RuntimeError(f"Unknown binary operator: {op}")
        
    def evaluate_unary_op(self, node: UnaryOpNode, env: Environment) -> Any:
        """Evaluate unary operations"""
        operand = self.evaluate(node.operand, env)
        op = node.operator
        
        if op == TokenType.MINUS:
            return -operand
        if op == TokenType.SUS:
            return not self.is_truthy(operand)
            
        raise RuntimeError(f"Unknown unary operator: {op}")
        
    def call_function(self, node: FunctionCallNode, env: Environment) -> Any:
        """Call a function"""
        # Built-in functions
        if node.name == 'goat':  # max
            args = [self.evaluate(arg, env) for arg in node.args]
            return max(args)
        if node.name == 'simp':  # min
            args = [self.evaluate(arg, env) for arg in node.args]
            return min(args)
        if node.name == 'chad':  # abs
            args = [self.evaluate(arg, env) for arg in node.args]
            return abs(args[0])
            
        # User-defined functions
        func = env.get(node.name)
        if not isinstance(func, Function):
            raise TypeError(f"{node.name} is not a function")
            
        if len(node.args) != len(func.params):
            raise TypeError(f"{func.name} expects {len(func.params)} arguments, got {len(node.args)}")
            
        # Create new environment for function execution
        func_env = Environment(func.closure)
        for param, arg in zip(func.params, node.args):
            func_env.define(param, self.evaluate(arg, env))
            
        try:
            self.evaluate(func.body, func_env)
            return None
        except ReturnException as e:
            return e.value
            
    def is_truthy(self, value: Any) -> bool:
        """Determine if a value is truthy"""
        if isinstance(value, bool):
            return value
        if value is None:
            return False
        if isinstance(value, (int, float)):
            return value != 0
        if isinstance(value, str):
            return len(value) > 0
        if isinstance(value, list):
            return len(value) > 0
        return True


# ============================================================================
# GRAPHICAL USER INTERFACE
# ============================================================================

class GZAllIDE:
    """Integrated Development Environment for GZAll"""
    
    def __init__(self, root):
        self.root = root
        self.root.title("GZAll IDE - Gen Z & Alpha Programming Language")
        self.root.geometry("1200x800")
        
        # Configure style
        self.style = ttk.Style()
        self.style.theme_use('clam')
        
        self.setup_ui()
        self.interpreter = None
        
    def setup_ui(self):
        """Setup the user interface"""
        # Main container
        main_frame = ttk.Frame(self.root, padding="10")
        main_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        
        # Configure grid weights
        self.root.columnconfigure(0, weight=1)
        self.root.rowconfigure(0, weight=1)
        main_frame.columnconfigure(0, weight=1)
        main_frame.rowconfigure(1, weight=1)
        
        # Title and info
        title_frame = ttk.Frame(main_frame)
        title_frame.grid(row=0, column=0, sticky=(tk.W, tk.E), pady=(0, 10))
        
        title_label = ttk.Label(
            title_frame, 
            text="🔥 GZAll - Gen Z & Alpha Programming Language 🔥",
            font=('Arial', 16, 'bold')
        )
        title_label.pack()
        
        subtitle_label = ttk.Label(
            title_frame,
            text="Created by the President Honourable of the Gen Z and A Population, all rights reserved to the sweed",
            font=('Arial', 9, 'italic')
        )
        subtitle_label.pack()
        
        # Create notebook for tabs
        self.notebook = ttk.Notebook(main_frame)
        self.notebook.grid(row=1, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        
        # Editor tab
        self.create_editor_tab()
        
        # Documentation tab
        self.create_docs_tab()
        
        # Examples tab
        self.create_examples_tab()
        
    def create_editor_tab(self):
        """Create the code editor tab"""
        editor_frame = ttk.Frame(self.notebook)
        self.notebook.add(editor_frame, text="💻 Code Editor")
        
        editor_frame.columnconfigure(0, weight=1)
        editor_frame.rowconfigure(0, weight=1)
        
        # Create paned window for editor and output
        paned = ttk.PanedWindow(editor_frame, orient=tk.VERTICAL)
        paned.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        
        # Code editor
        editor_container = ttk.Frame(paned)
        editor_label = ttk.Label(editor_container, text="Code:", font=('Arial', 10, 'bold'))
        editor_label.pack(anchor=tk.W, padx=5, pady=5)
        
        self.code_editor = scrolledtext.ScrolledText(
            editor_container,
            width=80,
            height=20,
            font=('Consolas', 11),
            wrap=tk.WORD,
            undo=True
        )
        self.code_editor.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        paned.add(editor_container, weight=2)
        
        # Output area
        output_container = ttk.Frame(paned)
        output_label = ttk.Label(output_container, text="Output:", font=('Arial', 10, 'bold'))
        output_label.pack(anchor=tk.W, padx=5, pady=5)
        
        self.output_area = scrolledtext.ScrolledText(
            output_container,
            width=80,
            height=10,
            font=('Consolas', 10),
            wrap=tk.WORD,
            state=tk.DISABLED
        )
        self.output_area.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        paned.add(output_container, weight=1)
        
        # Button frame
        button_frame = ttk.Frame(editor_frame)
        button_frame.grid(row=1, column=0, pady=10)
        
        run_button = ttk.Button(
            button_frame,
            text="▶ Run Code",
            command=self.run_code
        )
        run_button.pack(side=tk.LEFT, padx=5)
        
        clear_button = ttk.Button(
            button_frame,
            text="🗑 Clear Output",
            command=self.clear_output
        )
        clear_button.pack(side=tk.LEFT, padx=5)
        
        clear_editor_button = ttk.Button(
            button_frame,
            text="📄 New File",
            command=self.clear_editor
        )
        clear_editor_button.pack(side=tk.LEFT, padx=5)
        
    def create_docs_tab(self):
        """Create the documentation tab"""
        docs_frame = ttk.Frame(self.notebook)
        self.notebook.add(docs_frame, text="📚 Documentation")
        
        docs_text = scrolledtext.ScrolledText(
            docs_frame,
            width=80,
            height=30,
            font=('Consolas', 10),
            wrap=tk.WORD
        )
        docs_text.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        documentation = """
🔥 GZAll Programming Language Documentation 🔥

=== KEYWORDS ===

nocap          - Print output (no cap = truth)
fr             - Declare variable (for real)
bussin         - Define function 
slay           - Return from function
vibe           - If statement
bruh           - Else if statement
midcheck       - Else statement
bet            - While loop
lowkey         - For loop
deadass        - Boolean true
cap            - Boolean false
highkey        - Logical AND
salty          - Logical OR
sus            - Logical NOT
ghost          - Break from loop
rizz           - Get user input
slaps          - Assignment operator
hits           - Multiply
ratio          - Divide
mood           - Equals (==)
aint           - Not equals (!=)
bigger         - Greater than
smaller        - Less than
goat           - Max function
simp           - Min function
chad           - Absolute value
sheesh         - Comment (rest of line ignored)

=== SYNTAX ===

1. Variables:
   fr variableName slaps value
   Example: fr age slaps 18

2. Print:
   nocap expression1, expression2
   Example: nocap "Hello World!"

3. Input:
   fr name slaps rizz("Enter name: ")

4. If/Else:
   vibe condition:
       statements
   bruh other_condition:
       statements
   midcheck:
       statements

5. While Loop:
   bet condition:
       statements

6. For Loop:
   lowkey item in [1, 2, 3]:
       statements

7. Functions:
   bussin functionName(param1, param2):
       statements
       slay result

8. Comments:
   sheesh This is a comment

=== OPERATORS ===

Arithmetic: +, -, *, /, %
Comparison: mood (==), aint (!=), bigger (>), smaller (<)
Logical: highkey (and), salty (or), sus (not)
Assignment: slaps (=)

=== EXAMPLES ===

1. Hello World:
   nocap "Hello, World! No cap fr fr"

2. Variables:
   fr age slaps 21
   fr name slaps "Chad"
   nocap name, "is", age, "years old"

3. Conditionals:
   fr score slaps 95
   vibe score bigger 90:
       nocap "That's bussin!"
   midcheck:
       nocap "That's mid"

4. Loops:
   fr counter slaps 0
   bet counter smaller 5:
       nocap counter
       counter slaps counter + 1

5. Functions:
   bussin greet(name):
       nocap "Yo", name
       slay "What's good?"
   
   fr message slaps greet("homie")
   nocap message

6. Lists:
   fr vibes slaps [1, 2, 3, 4, 5]
   nocap goat(vibes)  sheesh prints 5
   nocap simp(vibes)  sheesh prints 1

=== TIPS ===

- Use 'sheesh' for comments
- 'deadass' = true, 'cap' = false
- Chain comparisons with 'highkey' (and) or 'salty' (or)
- Use 'ghost' to break out of loops
- Built-in functions: goat (max), simp (min), chad (abs)

Stay based, keep it 100! 💯
"""
        docs_text.insert('1.0', documentation)
        docs_text.config(state=tk.DISABLED)
        
    def create_examples_tab(self):
        """Create the examples tab"""
        examples_frame = ttk.Frame(self.notebook)
        self.notebook.add(examples_frame, text="📝 Examples")
        
        examples_frame.columnconfigure(0, weight=1)
        examples_frame.rowconfigure(0, weight=1)
        
        # Listbox for example selection
        list_frame = ttk.Frame(examples_frame)
        list_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S), padx=10, pady=10)
        
        list_label = ttk.Label(list_frame, text="Select an example:", font=('Arial', 10, 'bold'))
        list_label.pack(anchor=tk.W, pady=5)
        
        self.examples_list = tk.Listbox(list_frame, font=('Arial', 10))
        self.examples_list.pack(fill=tk.BOTH, expand=True)
        
        self.examples = {
            "Hello World": 'nocap "Hello, World! No cap fr fr"',
            
            "Variables and Math": '''fr x slaps 10
fr y slaps 20
fr sum slaps x + y
nocap "Sum is:", sum''',
            
            "Conditionals": '''fr age slaps 18
vibe age bigger 21:
    nocap "You can hit the club!"
bruh age mood 18:
    nocap "Just became an adult!"
midcheck:
    nocap "Still young fr"''',
            
            "While Loop": '''fr count slaps 1
bet count smaller 6:
    nocap "Count:", count
    count slaps count + 1''',
            
            "For Loop with List": '''fr numbers slaps [1, 2, 3, 4, 5]
lowkey num in numbers:
    nocap num hits 2''',
            
            "Functions": '''bussin add(a, b):
    slay a + b

fr result slaps add(5, 3)
nocap "Result:", result''',
            
            "User Input": '''fr name slaps rizz("What's your name? ")
nocap "Yo", name, "! Welcome to GZAll!"''',
            
            "Max and Min": '''fr nums slaps [10, 50, 30, 90, 20]
nocap "Max:", goat(nums)
nocap "Min:", simp(nums)''',
        }
        
        for example_name in self.examples.keys():
            self.examples_list.insert(tk.END, example_name)
            
        load_button = ttk.Button(
            list_frame,
            text="Load Example",
            command=self.load_example
        )
        load_button.pack(pady=10)
        
    def load_example(self):
        """Load selected example into editor"""
        selection = self.examples_list.curselection()
        if not selection:
            messagebox.showwarning("No Selection", "Please select an example first!")
            return
            
        example_name = self.examples_list.get(selection[0])
        example_code = self.examples.get(example_name, "")
        
        self.code_editor.delete('1.0', tk.END)
        self.code_editor.insert('1.0', example_code)
        self.notebook.select(0)  # Switch to editor tab
        
    def run_code(self):
        """Execute the code in the editor"""
        code = self.code_editor.get('1.0', tk.END)
        self.clear_output()
        
        try:
            # Tokenize
            lexer = Lexer(code)
            tokens = lexer.tokenize()
            
            # Parse
            parser = Parser(tokens)
            ast = parser.parse()
            
            # Interpret
            output_buffer = io.StringIO()
            
            def output_callback(text):
                output_buffer.write(str(text) + '\n')
                
            def input_callback(prompt=''):
                # Create a simple input dialog
                return tk.simpledialog.askstring("Input", prompt or "Enter value:")
                
            self.interpreter = Interpreter(output_callback, input_callback)
            self.interpreter.interpret(ast)
            
            # Display output
            output = output_buffer.getvalue()
            if output:
                self.append_output(output)
            else:
                self.append_output("Program executed successfully (no output)")
                
        except Exception as e:
            self.append_output(f"❌ Error: {str(e)}", 'error')
            
    def append_output(self, text, tag=None):
        """Append text to output area"""
        self.output_area.config(state=tk.NORMAL)
        self.output_area.insert(tk.END, text + '\n')
        if tag:
            # Could add colored tags here
            pass
        self.output_area.see(tk.END)
        self.output_area.config(state=tk.DISABLED)
        
    def clear_output(self):
        """Clear the output area"""
        self.output_area.config(state=tk.NORMAL)
        self.output_area.delete('1.0', tk.END)
        self.output_area.config(state=tk.DISABLED)
        
    def clear_editor(self):
        """Clear the code editor"""
        self.code_editor.delete('1.0', tk.END)


# ============================================================================
# MAIN ENTRY POINT
# ============================================================================

def main():
    """Main entry point for GZAll IDE"""
    import tkinter.simpledialog
    
    root = tk.Tk()
    app = GZAllIDE(root)
    root.mainloop()


if __name__ == "__main__":
    main()
