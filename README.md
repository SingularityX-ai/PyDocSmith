The project is a fork from the original project - https://github.com/rr-/docstring_parser/

I have updated it based on use case for - https://www.snorkell.ai/

# PyDocSmith

PyDocSmith is a versatile Python package designed for parsing, detecting, and composing docstrings in various styles. It supports multiple docstring conventions, including reStructuredText (reST), Google, NumPydoc, and Epydoc, providing flexibility in documentation practices for Python developers.

## Features

- **Docstring Style Detection:** Automatically detect the style of docstrings (e.g., reST, Google, NumPydoc, Epydoc) using simple heuristics.
- **Docstring Parsing:** Convert docstrings into structured representations, making it easier to analyze and manipulate documentation.
- **Docstring Composition:** Render structured docstrings back into text, allowing for automated docstring generation and modification.
- **Attribute Docstrings:** Parse attribute docstrings defined at class and module levels, enhancing the documentation of class properties and module-level variables.

## Installation

```bash
pip install PyDocSmith
```

## Usage

### Detecting Docstring Style

Detect the docstring style of a given text:

```python
from PyDocSmith import detect_docstring_style, DocstringStyle

docstring = """
This is an example docstring.
:param param1: Description of param1
:return: Description of return value
"""
style = detect_docstring_style(docstring)
print(style)  # Outputs: DocstringStyle.EPYDOC
```

### Parsing Docstrings

Parse a docstring into its components:

```python
from PyDocSmith import parse, DocstringStyle

parsed_docstring = parse(docstring, style=DocstringStyle.AUTO)
print(parsed_docstring)
```

### Composing Docstrings

Render a parsed docstring back into text:

```python
from PyDocSmith import compose

docstring_text = compose(parsed_docstring, style=DocstringStyle.REST)
print(docstring_text)
```

## Advanced Features

- **Parse From Object:** PyDocSmith can parse docstrings directly from Python objects, including classes and modules, incorporating attribute docstrings into the structured representation.
- **Custom Rendering Styles:** Customize the rendering of docstrings with compact or detailed styles, and specify custom indentation for the generated docstring text.

## Things that have been modified wrt to docstring_parser

1. Better heuristics to detect docstring style
2. Google Docstring has been modified to accommodate Notes, Examples
3. Sometime GoogleDoc string doesn't have proper indentation specially when generated from LLMs like GPT or Mistral. PyDocSmith can fix those bad docstrings.
4. Additional test-cases were added to accommodate a different style of GoogleDocstring

## Contributing

Contributions are welcome! Please submit pull requests or report issues on the project's GitHub page.

