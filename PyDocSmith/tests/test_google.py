"""Tests for Google-style docstring routines."""

import typing as T

import pytest
from PyDocSmith.common import Docstring, ParseError, RenderingStyle
from PyDocSmith.google import (
    GoogleParser,
    Section,
    SectionType,
    compose,
    parse,
)


def test_google_parser_unknown_section() -> None:
    """Test parsing an unknown section with default GoogleParser
    configuration.
    """
    parser = GoogleParser()
    docstring = parser.parse(
        """
        Unknown:
            spam: a
        """
    )
    assert docstring.short_description == "Unknown:"
    assert docstring.long_description == "spam: a"
    assert len(docstring.meta) == 0


def test_google_parser_custom_sections() -> None:
    """Test parsing an unknown section with custom GoogleParser
    configuration.
    """
    parser = GoogleParser(
        [
            Section("DESCRIPTION", "desc", SectionType.SINGULAR),
            Section("ARGUMENTS", "param", SectionType.MULTIPLE),
            Section("ATTRIBUTES", "attribute", SectionType.MULTIPLE),
            Section("EXAMPLES", "examples", SectionType.SINGULAR),
        ],
        title_colon=False,
    )
    docstring = parser.parse(
        """
        DESCRIPTION
            This is the description.

        ARGUMENTS
            arg1: first arg
            arg2: second arg

        ATTRIBUTES
            attr1: first attribute
            attr2: second attribute

        EXAMPLES
            Many examples
            More examples
        """
    )

    assert docstring.short_description is None
    assert docstring.long_description is None
    assert len(docstring.meta) == 6
    assert docstring.meta[0].args == ["desc"]
    assert docstring.meta[0].description == "This is the description."
    assert docstring.meta[1].args == ["param", "arg1"]
    assert docstring.meta[1].description == "first arg"
    assert docstring.meta[2].args == ["param", "arg2"]
    assert docstring.meta[2].description == "second arg"
    assert docstring.meta[3].args == ["attribute", "attr1"]
    assert docstring.meta[3].description == "first attribute"
    assert docstring.meta[4].args == ["attribute", "attr2"]
    assert docstring.meta[4].description == "second attribute"
    assert docstring.meta[5].args == ["examples"]
    assert docstring.meta[5].description == "Many examples\nMore examples"


def test_google_parser_custom_sections_after() -> None:
    """Test parsing an unknown section with custom GoogleParser configuration
    that was set at a runtime.
    """
    parser = GoogleParser(title_colon=False)
    parser.add_section(Section("Note", "note", SectionType.SINGULAR))
    docstring = parser.parse(
        """
        short description

        Note:
            a note
        """
    )
    assert docstring.short_description == "short description"
    assert docstring.long_description == "Note:\n    a note"

    docstring = parser.parse(
        """
        short description

        Note a note
        """
    )
    assert docstring.short_description == "short description"
    assert docstring.long_description == "Note a note"

    docstring = parser.parse(
        """
        short description

        Note
            a note
        """
    )
    assert len(docstring.meta) == 1
    assert docstring.meta[0].args == ["note"]
    assert docstring.meta[0].description == "a note"


@pytest.mark.parametrize(
    "source, expected",
    [
        ("", None),
        ("\n", None),
        ("Short description", "Short description"),
        ("\nShort description\n", "Short description"),
        ("\n   Short description\n", "Short description"),
    ],
)
def test_short_description(source: str, expected: str) -> None:
    """Test parsing short description."""
    docstring = parse(source)
    assert docstring.short_description == expected
    assert docstring.long_description is None
    assert not docstring.meta


@pytest.mark.parametrize(
    "source, expected_short_desc, expected_long_desc, expected_blank",
    [
        (
            "Short description\n\nLong description",
            "Short description",
            "Long description",
            True,
        ),
        (
            """
            Short description

            Long description
            """,
            "Short description",
            "Long description",
            True,
        ),
        (
            """
            Short description

            Long description
            Second line
            """,
            "Short description",
            "Long description\nSecond line",
            True,
        ),
        (
            "Short description\nLong description",
            "Short description",
            "Long description",
            False,
        ),
        (
            """
            Short description
            Long description
            """,
            "Short description",
            "Long description",
            False,
        ),
        (
            "\nShort description\nLong description\n",
            "Short description",
            "Long description",
            False,
        ),
        (
            """
            Short description
            Long description
            Second line
            """,
            "Short description",
            "Long description\nSecond line",
            False,
        ),
    ],
)
def test_long_description(
    source: str,
    expected_short_desc: str,
    expected_long_desc: str,
    expected_blank: bool,
) -> None:
    """Test parsing long description."""
    docstring = parse(source)
    assert docstring.short_description == expected_short_desc
    assert docstring.long_description == expected_long_desc
    assert docstring.blank_after_short_description == expected_blank
    assert not docstring.meta


@pytest.mark.parametrize(
    "source, expected_short_desc, expected_long_desc, "
    "expected_blank_short_desc, expected_blank_long_desc",
    [
        (
            """
            Short description
            Args:
                asd:
            """,
            "Short description",
            None,
            False,
            False,
        ),
        (
            """
            Short description
            Long description
            Args:
                asd:
            """,
            "Short description",
            "Long description",
            False,
            False,
        ),
        (
            """
            Short description
            First line
                Second line
            Args:
                asd:
            """,
            "Short description",
            "First line\n    Second line",
            False,
            False,
        ),
        (
            """
            Short description

            First line
                Second line
            Args:
                asd:
            """,
            "Short description",
            "First line\n    Second line",
            True,
            False,
        ),
        (
            """
            Short description

            First line
                Second line

            Args:
                asd:
            """,
            "Short description",
            "First line\n    Second line",
            True,
            True,
        ),
        (
            """
            Args:
                asd:
            """,
            None,
            None,
            False,
            False,
        ),
    ],
)
def test_meta_newlines(
    source: str,
    expected_short_desc: T.Optional[str],
    expected_long_desc: T.Optional[str],
    expected_blank_short_desc: bool,
    expected_blank_long_desc: bool,
) -> None:
    """Test parsing newlines around description sections."""
    docstring = parse(source)
    assert docstring.short_description == expected_short_desc
    assert docstring.long_description == expected_long_desc
    assert docstring.blank_after_short_description == expected_blank_short_desc
    assert docstring.blank_after_long_description == expected_blank_long_desc
    assert len(docstring.meta) == 1


def test_meta_with_multiline_description() -> None:
    """Test parsing multiline meta documentation."""
    docstring = parse(
        """
        Short description

        Args:
            spam: asd
                1
                    2
                3
        """
    )
    assert docstring.short_description == "Short description"
    assert len(docstring.meta) == 1
    assert docstring.meta[0].args == ["param", "spam"]
    assert docstring.meta[0].arg_name == "spam"
    assert docstring.meta[0].description == "asd\n1\n    2\n3"


def test_default_args() -> None:
    """Test parsing default arguments."""
    docstring = parse(
        """A sample function

        A function the demonstrates docstrings

        Args:
            arg1 (int): The firsty arg
            arg2 (str): The second arg
            arg3 (float, optional): The third arg. Defaults to 1.0.
            arg4 (Optional[Dict[str, Any]], optional): The last arg. Defaults to None.
            arg5 (str, optional): The fifth arg. Defaults to DEFAULT_ARG5.

        Returns:
            Mapping[str, Any]: The args packed in a mapping
        """
    )
    assert docstring is not None
    assert len(docstring.params) == 5

    arg4 = docstring.params[3]
    assert arg4.arg_name == "arg4"
    assert arg4.is_optional
    assert arg4.type_name == "Optional[Dict[str, Any]]"
    assert arg4.default == "None"
    assert arg4.description == "The last arg. Defaults to None."


def test_mixed_return_type() -> None:
    """Test parsing mixed return type."""
    docstring = parse(
        """        Generate output text based on output_ids, input_lengths, max_output_len, and tokenizer.
        Args:
            output_ids (Tensor): The output ids generated by the model.
            input_lengths (Tensor): The lengths of the input sequences.
            max_output_len (int): The maximum length of the output text.
            tokenizer (Tokenizer): The tokenizer used to decode the output ids.
        Returns:
            output_text (str): The decoded output text.
            - outputs (list): The list of output ids after removing extra eos ids.
        """
    )
    print(compose(docstring))
    assert (
        docstring.short_description
        == "Generate output text based on output_ids, input_lengths, max_output_len, and tokenizer."
    )
    assert len(docstring.params) == 4
    assert docstring.returns is not None
    assert docstring.many_returns is not None
    assert len(docstring.many_returns) == 2


def test_multiple_meta2() -> None:
    """Test parsing multiple meta."""
    docstring = parse(
        """
        Short description

        Args:
            spam: asd
                1
                    2
                3

        Raises:
            bla: herp
            yay: derp
        """
    )
    assert docstring.short_description == "Short description"
    assert len(docstring.meta) == 3
    assert docstring.meta[0].args == ["param", "spam"]
    assert docstring.meta[0].arg_name == "spam"
    assert docstring.meta[0].description == "asd\n1\n    2\n3"
    assert docstring.meta[1].args == ["raises", "bla"]
    assert docstring.meta[1].type_name == "bla"
    assert docstring.meta[1].description == "herp"
    assert docstring.meta[2].args == ["raises", "yay"]
    assert docstring.meta[2].type_name == "yay"
    assert docstring.meta[2].description == "derp"


def test_multiple_meta() -> None:
    """Test parsing multiple meta."""
    docstring = parse(
        """
        Short description

        Args:
            spam: asd
                1
                    2
                3

        Raises:
            bla: herp
            yay: derp
        """
    )
    assert docstring.short_description == "Short description"
    assert len(docstring.meta) == 3
    assert docstring.meta[0].args == ["param", "spam"]
    assert docstring.meta[0].arg_name == "spam"
    assert docstring.meta[0].description == "asd\n1\n    2\n3"
    assert docstring.meta[1].args == ["raises", "bla"]
    assert docstring.meta[1].type_name == "bla"
    assert docstring.meta[1].description == "herp"
    assert docstring.meta[2].args == ["raises", "yay"]
    assert docstring.meta[2].type_name == "yay"
    assert docstring.meta[2].description == "derp"


def test_params() -> None:
    """Test parsing params."""
    docstring = parse("Short description")
    assert len(docstring.params) == 0

    docstring = parse(
        """
        Short description

        Args:
            name: description 1
            priority (int): description 2
            sender (str?): description 3
            ratio (Optional[float], optional): description 4
        """
    )
    assert len(docstring.params) == 4
    assert docstring.params[0].arg_name == "name"
    assert docstring.params[0].type_name is None
    assert docstring.params[0].description == "description 1"
    assert not docstring.params[0].is_optional
    assert docstring.params[1].arg_name == "priority"
    assert docstring.params[1].type_name == "int"
    assert docstring.params[1].description == "description 2"
    assert not docstring.params[1].is_optional
    assert docstring.params[2].arg_name == "sender"
    assert docstring.params[2].type_name == "str"
    assert docstring.params[2].description == "description 3"
    assert docstring.params[2].is_optional
    assert docstring.params[3].arg_name == "ratio"
    assert docstring.params[3].type_name == "Optional[float]"
    assert docstring.params[3].description == "description 4"
    assert docstring.params[3].is_optional

    docstring = parse(
        """
        Short description

        Args:
            name: description 1
                with multi-line text
            priority (int): description 2
        """
    )
    assert len(docstring.params) == 2
    assert docstring.params[0].arg_name == "name"
    assert docstring.params[0].type_name is None
    assert docstring.params[0].description == (
        "description 1\nwith multi-line text"
    )
    assert docstring.params[1].arg_name == "priority"
    assert docstring.params[1].type_name == "int"
    assert docstring.params[1].description == "description 2"


def test_attributes() -> None:
    """Test parsing attributes."""
    docstring = parse("Short description")
    assert len(docstring.params) == 0

    docstring = parse(
        """
        Short description

        Attributes:
            name: description 1
            priority (int): description 2
            sender (str?): description 3
            ratio (Optional[float], optional): description 4
        """
    )
    assert len(docstring.params) == 4
    assert docstring.params[0].arg_name == "name"
    assert docstring.params[0].type_name is None
    assert docstring.params[0].description == "description 1"
    assert not docstring.params[0].is_optional
    assert docstring.params[1].arg_name == "priority"
    assert docstring.params[1].type_name == "int"
    assert docstring.params[1].description == "description 2"
    assert not docstring.params[1].is_optional
    assert docstring.params[2].arg_name == "sender"
    assert docstring.params[2].type_name == "str"
    assert docstring.params[2].description == "description 3"
    assert docstring.params[2].is_optional
    assert docstring.params[3].arg_name == "ratio"
    assert docstring.params[3].type_name == "Optional[float]"
    assert docstring.params[3].description == "description 4"
    assert docstring.params[3].is_optional

    docstring = parse(
        """
        Short description

        Attributes:
            name: description 1
                with multi-line text
            priority (int): description 2
        """
    )
    assert len(docstring.params) == 2
    assert docstring.params[0].arg_name == "name"
    assert docstring.params[0].type_name is None
    assert docstring.params[0].description == (
        "description 1\nwith multi-line text"
    )
    assert docstring.params[1].arg_name == "priority"
    assert docstring.params[1].type_name == "int"
    assert docstring.params[1].description == "description 2"


def test_returns() -> None:
    """Test parsing returns. It's failing"""
    docstring = parse(
        """
        Short description
        """
    )
    assert docstring.returns is None
    assert docstring.many_returns is not None
    assert len(docstring.many_returns) == 0

    docstring = parse(
        """
        Short description
        Returns:
            description
        """
    )
    assert docstring.returns is not None
    assert docstring.returns.type_name is None
    assert docstring.returns.description == "description"
    assert docstring.many_returns is not None
    assert len(docstring.many_returns) == 1
    assert docstring.many_returns[0] == docstring.returns

    docstring = parse(
        """
        Short description
        Returns:
            description with: a colon!
        """
    )
    assert docstring.returns is not None
    assert docstring.returns.type_name is None
    assert docstring.returns.description == "description with: a colon!"
    assert docstring.many_returns is not None
    assert len(docstring.many_returns) == 1
    assert docstring.many_returns[0] == docstring.returns

    docstring = parse(
        """
        Short description
        Returns:
            int: description
        """
    )
    assert docstring.returns is not None
    assert docstring.returns.type_name == "int"
    assert docstring.returns.description == "description"
    assert docstring.many_returns is not None
    assert len(docstring.many_returns) == 1
    assert docstring.many_returns[0] == docstring.returns

    docstring = parse(
        """
        Returns:
            Optional[Mapping[str, List[int]]]: A description: with a colon
        """
    )
    assert docstring.returns is not None
    assert docstring.returns.type_name == "Optional[Mapping[str, List[int]]]"
    assert docstring.returns.description == "A description: with a colon"
    assert docstring.many_returns is not None
    assert len(docstring.many_returns) == 1
    assert docstring.many_returns[0] == docstring.returns

    docstring = parse(
        """
        Short description
        Yields:
            int: description
        """
    )
    assert docstring.returns is not None
    assert docstring.returns.type_name == "int"
    assert docstring.returns.description == "description"
    assert docstring.many_returns is not None
    assert len(docstring.many_returns) == 1
    assert docstring.many_returns[0] == docstring.returns

    docstring = parse(
        """
        Short description
        Returns:
            int: description
            with much text

            even some spacing
        """
    )
    assert docstring.returns is not None
    assert docstring.returns.type_name == "int"
    assert docstring.returns.description == (
        "description\nwith much text\n\neven some spacing"
    )
    assert docstring.many_returns is not None
    assert len(docstring.many_returns) == 1
    assert docstring.many_returns[0] == docstring.returns


def test_raises() -> None:
    """Test parsing raises."""
    docstring = parse(
        """
        Short description
        """
    )
    assert len(docstring.raises) == 0

    docstring = parse(
        """
        Short description
        Raises:
            ValueError: description
        """
    )
    assert len(docstring.raises) == 1
    assert docstring.raises[0].type_name == "ValueError"
    assert docstring.raises[0].description == "description"


def test_examples() -> None:
    """Test parsing examples."""
    docstring = parse(
        """
        Short description
        Example:
            example: 1
        Examples:
            long example

            more here
        """
    )
    assert len(docstring.examples) == 2
    assert docstring.examples[0].description == "example: 1"
    assert docstring.examples[1].description == "long example\n\nmore here"


def test_parsing_logic() -> None:
    """Test parsing examples. Fix this test case by fixing the parse function in google.py"""
    docstring = parse(
        """
        Creates a task instance from its configuration.
        Args:
        - task_config (dict): Configuration for the task.
        Returns:
        - Task: Task instance created from the configuration.
        Raises:
        - StopIteration: If no agent with the specified role is found in the list of agents.
        """
    )
    assert (
        docstring.short_description
        == "Creates a task instance from its configuration."
    )
    assert len(docstring.params) == 1
    assert docstring.params[0].arg_name == "task_config"
    assert docstring.returns is not None
    assert (
        docstring.returns.description
        == "Task: Task instance created from the configuration."
    )  # TODO: Fix it,it should have arg_name
    assert (
        docstring.returns.type_name == "Task"
    )  # TODO: Fix it,it should have arg_name
    assert len(docstring.raises) == 1
    assert docstring.raises[0].type_name == "StopIteration"


def test_parsing_logic_2() -> None:
    """Test parsing examples."""
    docstring = parse(
        """
        Return a list of tools for delegating work and asking questions to co-workers.

        This method returns a list of Tool objects, each representing \
            a specific tool for delegating work or asking questions \
                to co-workers.

        Returns:
            list: A list of Tool objects, each representing a specific tool for delegating
                work or asking questions to co-workers.
        Raises:
            (if applicable)
        """
    )
    assert (
        docstring.short_description
        == "Return a list of tools for delegating work and asking questions to co-workers."
    )
    assert docstring.long_description.startswith(
        "This method returns a list of Tool objects, each representing"
    )

    # todo: fix it, long_description shouldn't contain new line
    assert len(docstring.params) == 0
    assert docstring.returns is not None
    assert docstring.returns.arg_name is None
    assert docstring.returns.type_name == "list"
    assert len(docstring.raises) == 0


def test_parsing_logic_3() -> None:
    """Test parsing examples."""
    docstring = parse(
        """
        Useful for when you need to multiply two numbers together.
        Args:
        numbers (str): A comma separated list of numbers of length \
            two, representing the two numbers you want to multiply together.
        Returns:
        float: The result of multiplying the two input numbers together.
        Raises:
        ValueError: If the input format is incorrect or if the input numbers are not valid.
        Example:
        >>> multiplier("2,3")
        6
        """
    )
    assert (
        docstring.short_description
        == "Useful for when you need to multiply two numbers together."
    )
    assert len(docstring.params) == 1
    assert docstring.returns is not None
    assert docstring.examples is not None
    assert docstring.examples[0].description == '>>> multiplier("2,3")\n6'


def test_notes() -> None:
    """Test parsing examples."""
    docstring = parse(
        """
        Initialize the model with provided parameters.

        Args:
        model_path (Optional[str]): Path to the model. Defaults to None.
        engine_name (Optional[str]): Name of the engine. Defaults to None.
        tokenizer_dir (Optional[str]): Directory for the tokenizer. Defaults to None.
        temperature (float): Temperature for token generation. Defaults to 0.1.
        max_new_tokens (int): Maximum number of new tokens. Defaults to DEFAULT_NUM_OUTPUTS.
        context_window (int): Context window size. Defaults to DEFAULT_CONTEXT_WINDOW.
        messages_to_prompt (Optional[Callable]): Function for prompting messages. Defaults to None.
        completion_to_prompt (Optional[Callable]): Function for prompting completions. Defaults to None.
        callback_manager (Optional[CallbackManager]): Manager for callbacks. \
            Defaults to None.
        generate_kwargs (Optional[Dict[str, Any]]): Additional keyword \
            arguments for generation. Defaults to None.
        model_kwargs (Optional[Dict[str, Any]]): Additional keyword arguments for the model. Defaults to None.
        verbose (bool): Verbosity flag. Defaults to False.

        Raises:
        ValueError: If the provided model path does not exist.

        Note:
        This function initializes the model with the provided parameters \
            and sets up the necessary configurations and resources for \
                token generation and model decoding.
        """
    )
    assert (
        docstring.short_description
        == "Initialize the model with provided parameters."
    )
    assert len(docstring.params) == 12
    assert docstring.returns is None
    assert docstring.notes is not None
    assert docstring.notes[0].description.startswith(
        "This function initializes the model with the provided"
    )


def test_multiple_returns() -> None:
    """Test parsing examples."""
    docstring = parse(
        """A sample function

        A function the demonstrates docstrings

        Args:
            arg1 (int): The firsty arg
            arg2 (str): The second arg
            arg3 (float, optional): The third arg. Defaults to 1.0.
            arg4 (Optional[Dict[str, Any]], optional): The last arg. Defaults to None.
            arg5 (str, optional): The fifth arg. Defaults to DEFAULT_ARG5.

        Returns:
            arg1 (Optional[Dict[str, Any]], optional): The args packed in a mapping
            arg2 (float, optional): The second arg
        """
    )
    assert docstring is not None
    assert len(docstring.params) == 5

    arg4 = docstring.params[3]
    assert arg4.arg_name == "arg4"
    assert arg4.is_optional
    assert arg4.type_name == "Optional[Dict[str, Any]]"
    assert arg4.default == "None"
    assert arg4.description == "The last arg. Defaults to None."
    assert docstring.many_returns is not None


def test_single_returns() -> None:
    """Test parsing examples."""
    docstring = parse(
        """A sample function

        A function the demonstrates docstrings

        Args:
            arg1 (int): The firsty arg
            arg2 (str): The second arg
            arg3 (float, optional): The third arg. Defaults to 1.0.
            arg4 (Optional[Dict[str, Any]], optional): The last arg. Defaults to None.
            arg5 (str, optional): The fifth arg. Defaults to DEFAULT_ARG5.

        Returns:
            The args packed in a mapping
        """
    )
    assert docstring is not None
    assert len(docstring.params) == 5
    print(compose(docstring))

    arg4 = docstring.params[3]
    assert arg4.arg_name == "arg4"
    assert arg4.is_optional
    assert arg4.type_name == "Optional[Dict[str, Any]]"
    assert arg4.default == "None"
    assert arg4.description == "The last arg. Defaults to None."
    assert docstring.many_returns is not None


def test_none_returns() -> None:
    """Test parsing examples."""
    docstring = parse(
        """A sample function

        A function the demonstrates docstrings

        Args:
            arg1 (int): The firsty arg
            arg2 (str): The second arg
            arg3 (float, optional): The third arg. Defaults to 1.0.
            arg4 (Optional[Dict[str, Any]], optional): The last arg. Defaults to None.
            arg5 (str, optional): The fifth arg. Defaults to DEFAULT_ARG5.

        Returns:
            None
        """
    )
    new_docstring = compose(docstring, rendering_style=RenderingStyle.COMPACT)
    print(new_docstring)
    docstring = parse(new_docstring)
    assert docstring is not None
    assert len(docstring.params) == 5

    assert docstring.returns is None
    assert len(docstring.many_returns) == 0
    print(compose(docstring))


def test_broken_meta() -> None:
    """Test parsing broken meta."""
    with pytest.raises(ParseError):
        parse("Args:")

    z = parse("Args:\n    herp derp")
    assert len(z.params) == 0
    assert z.short_description is None


def test_unknown_meta() -> None:
    # currently failing
    """Test parsing unknown meta. This is failing"""
    docstring = parse(
        """Short desc

        Unknown 0:
            title0: content0

        Args:
            arg0: desc0
            arg1: desc1

        Unknown1:
            title1: content1

        Unknown2:
            title2: content2
        """
    )

    assert docstring.params[0].arg_name == "arg0"
    assert docstring.params[0].description == "desc0"
    assert docstring.params[1].arg_name == "arg1"
    assert docstring.params[1].description == "desc1"


def test_unformatted_valid_docstring() -> None:
    """
    Test parsing a valid docstring without formatting.
    """

    docstring = parse(
        """
    Set the RPM controller for the object.

    Args:
    rpm_controller: The RPM controller to be set.

    Raises:
    helo: gejjj

    Returns:
    None
    """
    )
    assert (
        docstring.short_description == "Set the RPM controller for the object."
    )
    assert len(docstring.params) == 1
    assert docstring.params[0].arg_name == "rpm_controller"
    assert len(docstring.raises) == 1
    assert docstring.raises[0].type_name == "helo"
    assert docstring.raises[0].description == "gejjj"
    assert docstring.returns is not None


def test_broken_arguments() -> None:
    """Test parsing broken arguments."""

    z: Docstring = parse(
        """This is a test

        Args:
            param - poorly formatted
        """
    )
    assert len(z.params) == 0


def test_empty_example() -> None:
    """Test parsing empty examples section."""
    docstring = parse(
        """Short description

        Example:

        Raises:
            IOError: some error
        """
    )

    assert len(docstring.examples) == 0


@pytest.mark.parametrize(
    "source, expected",
    [
        ("", ""),
        ("\n", ""),
        ("Short description", "Short description"),
        ("\nShort description\n", "Short description"),
        ("\n   Short description\n", "Short description"),
        (
            "Short description\n\nLong description",
            "Short description\n\nLong description",
        ),
        (
            """
            Short description

            Long description
            """,
            "Short description\n\nLong description",
        ),
        (
            """
            Short description

            Long description
            Second line
            """,
            "Short description\n\nLong description\nSecond line",
        ),
        (
            "Short description\nLong description",
            "Short description\nLong description",
        ),
        (
            """
            Short description
            Long description
            """,
            "Short description\nLong description",
        ),
        (
            "\nShort description\nLong description\n",
            "Short description\nLong description",
        ),
        (
            """
            Short description
            Long description
            Second line
            """,
            "Short description\nLong description\nSecond line",
        ),
        (
            """
            Short description
            Meta:
                asd
            """,
            "Short description\nMeta:\n    asd",
        ),
        (
            """
            Short description
            Long description
            Meta:
                asd
            """,
            "Short description\nLong description\nMeta:\n    asd",
        ),
        (
            """
            Short description
            First line
                Second line
            Meta:
                asd
            """,
            "Short description\n"
            "First line\n"
            "    Second line\n"
            "Meta:\n"
            "    asd",
        ),
        (
            """
            Short description

            First line
                Second line
            Meta:
                asd
            """,
            "Short description\n"
            "\n"
            "First line\n"
            "    Second line\n"
            "Meta:\n"
            "    asd",
        ),
        (
            """
            Short description

            First line
                Second line

            Meta:
                asd
            """,
            "Short description\n"
            "\n"
            "First line\n"
            "    Second line\n"
            "\n"
            "Meta:\n"
            "    asd",
        ),
        (
            """
            Short description

            Meta:
                asd
                    1
                        2
                    3
            """,
            "Short description\n"
            "\n"
            "Meta:\n"
            "    asd\n"
            "        1\n"
            "            2\n"
            "        3",
        ),
        (
            """
            Short description

            Meta1:
                asd
                1
                    2
                3
            Meta2:
                herp
            Meta3:
                derp
            """,
            "Short description\n"
            "\n"
            "Meta1:\n"
            "    asd\n"
            "    1\n"
            "        2\n"
            "    3\n"
            "Meta2:\n"
            "    herp\n"
            "Meta3:\n"
            "    derp",
        ),
        (
            """
            Short description

            Args:
                name: description 1
                priority (int): description 2
                sender (str, optional): description 3
                message (str, optional): description 4, defaults to 'hello'
                multiline (str?):
                    long description 5,
                        defaults to 'bye'
            """,
            "Short description\n"
            "\n"
            "Args:\n"
            "    name: description 1\n"
            "    priority (int): description 2\n"
            "    sender (str?): description 3\n"
            "    message (str?): description 4, defaults to 'hello'\n"
            "    multiline (str?): long description 5,\n"
            "        defaults to 'bye'",
        ),
        (
            """
            Short description
            Raises:
                ValueError: description
            """,
            "Short description\nRaises:\n    ValueError: description",
        ),
    ],
)
def test_compose(source: str, expected: str) -> None:
    """Test compose in default mode."""
    assert compose(parse(source)) == expected


@pytest.mark.parametrize(
    "source, expected",
    [
        (
            """
            Short description

            Args:
                name: description 1
                priority (int): description 2
                sender (str, optional): description 3
                message (str, optional): description 4, defaults to 'hello'
                multiline (str?):
                    long description 5,
                        defaults to 'bye'
            """,
            "Short description\n"
            "\n"
            "Args:\n"
            "    name: description 1\n"
            "    priority (int): description 2\n"
            "    sender (str, optional): description 3\n"
            "    message (str, optional): description 4, defaults to 'hello'\n"
            "    multiline (str, optional): long description 5,\n"
            "        defaults to 'bye'",
        ),
    ],
)
def test_compose_clean(source: str, expected: str) -> None:
    """Test compose in clean mode."""
    assert (
        compose(parse(source), rendering_style=RenderingStyle.CLEAN)
        == expected
    )


@pytest.mark.parametrize(
    "source, expected",
    [
        (
            """
            Short description

            Args:
                name: description 1
                priority (int): description 2
                sender (str, optional): description 3
                message (str, optional): description 4, defaults to 'hello'
                multiline (str?):
                    long description 5,
                        defaults to 'bye'
            """,
            "Short description\n"
            "\n"
            "Args:\n"
            "    name:\n"
            "        description 1\n"
            "    priority (int):\n"
            "        description 2\n"
            "    sender (str, optional):\n"
            "        description 3\n"
            "    message (str, optional):\n"
            "        description 4, defaults to 'hello'\n"
            "    multiline (str, optional):\n"
            "        long description 5,\n"
            "        defaults to 'bye'",
        ),
    ],
)
def test_compose_expanded(source: str, expected: str) -> None:
    """Test compose in expanded mode."""
    assert (
        compose(parse(source), rendering_style=RenderingStyle.EXPANDED)
        == expected
    )
