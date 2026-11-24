"""Source text to tokens, using the original lexical rules."""
import re

KEYWORDS = {"if", "else", "while"}
# Captures one identifier/keyword, integer, or individual non-whitespace token.
TOKEN = re.compile(r"[A-Za-z_]\w*|\d+|\S")


def tokenize(source):
    return TOKEN.findall(source)


def is_operand(token):
    return token.isdigit() or (token.isidentifier() and token not in KEYWORDS)
