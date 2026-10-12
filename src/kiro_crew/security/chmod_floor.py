"""The argv-structural floor under the ``local-destructive-chmod-777`` row.

The catalog row is the literal ``chmod 777.*``: it recognises one SPELLING of the
world-rwx mode change, so any option between the program and the mode -- ``chmod
-R 777 ~``, a packed ``-Rv``, a split ``-R -v``, a long ``--recursive`` -- runs the
identical permission change while the PreToolUse gate admits it.
A regex that admits "any run of options" is the wrong instrument: it either pins
a few option spellings (and the one nobody enumerated, ``-Rvfc``, still bypasses)
or opens a gap the ReDoS screens in ``denied_rules`` exist to refuse.

So the question is asked of the ARGV instead, the way the self-protection rows
are asked (``argv_floor``): for every
command in the input, including each nested shell payload, is ``chmod`` the
program that runs, and does its MODE operand -- read the way the shell and then
``chmod`` read it, then evaluated with ``chmod``'s own mode grammar -- grant all
nine permission bits?  Options are recognised by shape at every position --
GNU ``chmod`` permutes them, so ``chmod 777 -R ~`` and ``chmod -R 777 ~`` are one
command -- however many options precede the mode and however they are packed.

It is a UNION with the regex row, never a replacement: the row stays in the
regex tier (so a text the tokenizer cannot see into is still caught, and a
tokenizer failure cannot fail OPEN), the floor runs only while the row is in the
effective set (an operator opt-out of the row disables both), and the catalog
pattern, the golden fixture and the governance pin map are untouched.

Every primitive here is one the package already relies on: the frame walk
(``_shell_payload_walk``) that descends ``bash -c``, ``eval``, heredocs and
substitutions to any depth; ``_argv_programs`` for "which command is this word
an argument OF"; ``_program_basename`` for the program a word invokes once
quoting, wrappers and redirects come off; and the shell reader's data-consumer
vocabulary to say which program a ``chmod`` word is an argument OF.  A word that
is a data consumer's argument (``grep -rn chmod src/``, ``rg -n chmod *``) is that
program's data and the words after it are that program's; a word under a program
that is NOT a data consumer -- ``sudo``, ``env``, ``nice``, ``xargs``,
``find -exec``, ``busybox``, a wrapper nobody enumerated -- or ``chmod`` itself
in program position is read as the invocation it is.

This module sits beside ``perm_verb_mention`` at the top of the package's
dependency order: it imports the reader and the floor, and nothing imports it
but the facade's evaluator (through ``_submodule``).  It lives in its own file
because ``argv_floor`` is held to a per-module line ceiling by
``test_no_single_module_grows_back_into_a_monolith``.
"""

import re

from .argv_floor import _brace_expansions
from .perm_verb_mention import _REDIRECT_AMP_RE
from .shell_normalizer import (
    _CONTROL_OPERATOR_RE,
    _DATA_CONSUMER_PROGRAMS,
    _NESTED_SHELL_PROGRAMS,
    _NESTED_SHELL_VERBS,
    _REDIRECT_START_RE,
    _argv_programs,
    _ends_argv,
    _heredoc_marker,
    _iter_shell_chars,
    _program_basename,
    _self_tokens,
    _shell_payload_walk,
)

#: The program this floor is about.  Compared against ``_program_basename`` of a
#: word, so ``/bin/chmod``, ``"chmod"``, ``ch''mod`` and ``chmod>/dev/null`` all
#: read as it, and ``--chmod=+x`` (a ``git update-index`` flag) does not.
_CHMOD_PROGRAM = "chmod"

#: The permission bits the row exists to refuse: read, write and execute for the
#: owner, the group AND everyone else.  A mode is a grant when the bits it leaves
#: set include all nine; the special-bits digit (``1777``, ``4777``) narrows
#: nothing.
_WORLD_RWX_BITS = 0o777

#: The largest value GNU ``mode_compile`` accepts for an octal number, wherever it
#: sits; ``77777`` and ``+77777`` are refused as invalid (measured).
_OCTAL_MODE_MAX = 0o7777

_OCTAL_DIGITS = frozenset("01234567")
_WHO_LETTERS = frozenset("ugoa")
_SYMBOLIC_PERMS = frozenset("rwxXstugo")
_OPERATORS = frozenset("+-=")


def _octal_run(text: str, start: int) -> int:
    """The index just past the run of octal digits that starts at *start*."""
    end = start
    while end < len(text) and text[end] in _OCTAL_DIGITS:
        end += 1
    return end


def _mode_grants_world_rwx(mode: str) -> bool:
    """True if GNU ``chmod`` applies *mode* as a NUMERIC grant of all nine bits.

    This is the mode grammar of ``mode_compile``, read rather than matched, so a
    respelling of the same value cannot step around it.  Each case below was
    measured against GNU coreutils' ``chmod`` on a file at mode ``000``:

    * a mode made only of octal digits is that value, with any run of leading
      zeros (``00777`` and ``000000777`` are ``777``) up to ``07777``;
    * otherwise the mode is a comma list of clauses, each an optional ``[ugoa]*``
      and one or more actions.  An action is an operator (``+``, ``-``, ``=``)
      followed by octal digits, symbolic letters, or nothing: ``+777`` adds the
      bits, ``=777`` sets them, ``-777`` removes them, and a list composes
      left to right (``=600,+777``, ``+07,+0770`` and ``-x+777`` all end at ``777``);
    * octal digits after a ``[ugoa]`` letter (``u=777``), a digit run straight
      after another (``+7+70``), an empty clause (``777,``) and a clause that is
      only digits inside a list (``u+x,777``) are INVALID, and an invalid mode
      changes nothing at all.

    Symbolic letters (``a+rwx``) are a different grammar, relative to the file's
    current bits and the umask, and the issue this floor closes leaves them alone,
    so a letter action contributes nothing here -- not even a removal.  Ignoring a
    removal can only over-read a grant (``=777,u-w`` is read as ``777``), the
    fail-closed direction; ignoring a letter GRANT is the documented symbolic
    residual (``u+rwx,+077``).

    BSD ``chmod`` (macOS) accepts a strict subset of these spellings, so a reading
    that is right for GNU cannot miss a grant there.
    """
    if mode and all(ch in _OCTAL_DIGITS for ch in mode):
        value = int(mode, 8)
        return value <= _OCTAL_MODE_MAX and value & _WORLD_RWX_BITS == _WORLD_RWX_BITS
    bits = 0
    for clause in mode.split(","):
        i = 0
        while i < len(clause) and clause[i] in _WHO_LETTERS:
            i += 1
        has_who = i > 0
        who_end = i
        if i == len(clause):
            return False  # empty clause, or a who list with no action: invalid
        while i < len(clause):
            op = clause[i]
            if op not in _OPERATORS:
                return False
            i += 1
            end = _octal_run(clause, i)
            if end > i:
                if has_who:
                    return False
                value = int(clause[i:end], 8)
                if value > _OCTAL_MODE_MAX:
                    return False
                if op == "+":
                    bits |= value
                elif op == "-":
                    bits &= ~value
                else:
                    bits = value
                i = end
                # GNU ends an action at its digit run: only a new clause may follow.
                if i < len(clause):
                    return False
                continue
            start = i
            while i < len(clause) and clause[i] in _SYMBOLIC_PERMS:
                i += 1
            bits &= ~_symbolic_narrowing(clause[:who_end], op, clause[start:i])
    return bits & _WORLD_RWX_BITS == _WORLD_RWX_BITS


_WHO_MASKS = {"u": 0o700, "g": 0o070, "o": 0o007, "a": 0o777}
_PERM_MASKS = {"r": 0o444, "w": 0o222, "x": 0o111}


def _symbolic_narrowing(who: str, op: str, letters: str) -> int:
    """The bits a symbolic action is CERTAIN to clear, for an explicit *who*.

    ``=777,go-w`` is ``755``, not a grant.  Only the part
    of a letter action that is independent of the file and the umask is applied,
    and only in the narrowing direction, so the floor never reads LESS grant than
    ``chmod`` applies except through a symbolic grant -- the residual the issue
    leaves alone:

    * with no who, ``chmod`` masks the action with the umask, which the gate does
      not know: nothing is certain, so nothing is cleared;
    * ``-`` clears the named ``r``/``w`` bits of each who;
    * ``=`` clears every ``r``/``w`` bit of each who the letters do not name;
    * ``x`` is never certain: the gate hands this floor LOWERCASED text, so it may
      be ``X`` (execute only for a directory or an already-executable file), and
      the copy letters ``u``/``g``/``o`` depend on the file -- so ``x`` is kept,
      and an action with a copy letter clears nothing;
    * ``+`` clears nothing.

    Measured against GNU ``chmod``: ``=777,go-w`` is ``755`` and ``=777,o=rx`` is
    ``775`` (both read exactly); ``=777,a-x`` is ``666`` (read as ``777``,
    over-strict).
    """
    if not who or op == "+":
        return 0
    if any(ch in "ugo" for ch in letters):
        return 0
    who_mask = 0
    for ch in who:
        who_mask |= _WHO_MASKS[ch]
    named = 0
    for ch in letters:
        named |= _PERM_MASKS.get(ch, 0)
    readable_writable = 0o666
    if op == "-":
        return who_mask & named & readable_writable
    return who_mask & ~named & readable_writable


#: The characters GNU ``chmod`` reads as the START of a mode when they follow a
#: leading ``-``: ``chmod -x f`` and ``chmod -w,+777 f`` hand the whole word to the
#: mode parser, and once any such word is seen there is no positional mode --
#: every operand is a file (GNU joins several such words with commas; BSD takes
#: one).  ``r`` is missing on purpose.  The gate hands this floor LOWERCASED text,
#: so ``-r`` is either the option ``-R`` or the removal ``-r``; reading it as the
#: option keeps the next operand as the mode, the fail-closed reading of the two.
#: A longer ``-r...`` word made only of mode characters is read as a mode (see
#: :func:`_is_dash_mode`).
_DASH_MODE_FIRST = frozenset("wxXstugoa,+=01234567")
_DASH_MODE_CHARS = frozenset("rwxXstugoa,+=01234567")


def _is_dash_mode(word: str) -> bool:
    """True if *word* is a ``-``-led MODE, not an option, as GNU ``chmod`` reads it.

    ``-x``, ``-w``, ``-777``, ``-+777`` and ``-x,+777`` are modes.  ``-R``, ``-v``,
    ``-Rv`` and every ``--long`` option are not.  A lowercased ``-r`` followed by
    only mode characters (``-rw``, ``-r,+777``) is a mode whether it was written
    ``-r`` or ``-R``: written ``-R``, the mode character after it makes GNU read the
    whole word as an (invalid) mode and change nothing, so evaluating it as a mode
    can only refuse.
    """
    if len(word) < 2 or word[0] != "-" or word[1] == "-":
        return False
    if word[1] in _DASH_MODE_FIRST:
        return True
    return word[1] == "r" and len(word) > 2 and all(ch in _DASH_MODE_CHARS for ch in word[2:])


#: Characters that make a word a pathname glob once braces are expanded.  A glob
#: in the MODE position is filled from the filesystem, which a static reader cannot
#: see, so it is refused (fail closed); nobody spells a mode as a glob on purpose.
#: A glob among the FILE operands is a filename and is not read.
_PATH_GLOB_RE = re.compile(r"[*?\[]")


def _expand_braces(word: str) -> "list[str] | None":
    """The words bash's brace expansion makes of *word*; None past the cap.

    The package's one expander, ``argv_floor._brace_expansions`` (alternatives,
    sequences, nesting, ``${...}`` left literal, a cap that fails closed).  It
    expands innermost groups first, so its word ORDER can differ from bash's for
    nested groups (``{{-v,777},644}`` comes back ``-v 644 777 644``; bash says
    ``-v 777 644``), and it drops a sequence's zero padding, which cannot change
    an octal value.  The caller therefore never relies on the order of a word that
    expands to several: see :func:`_brace_word_may_grant`.
    """
    if "{" not in word:
        return [word]
    return _brace_expansions(word)


def _brace_word_may_grant(words: "list[str]") -> bool:
    """True if any word one brace word expands to would grant, read as the mode.

    Asked only while the mode position is still open (no positional operand yet),
    so a brace group among the FILES (``chmod 755 {bin,scripts}/*.sh``) is not.
    Order-independent and fail closed: a brace word whose expansions include a
    world-rwx mode (``{00777,0755}``, ``{-x,-+777}``) is refused,
    including ``{755,777}``, whose first word is the mode ``755``.  Nobody spells a
    mode as a brace group by accident.
    """
    for word in words:
        # A dash-led mode word (``-+777``) is itself a mode-grammar string, so one
        # evaluation covers both the positional and the dash-led reading.
        if _PATH_GLOB_RE.search(word) or _mode_grants_world_rwx(word):
            return True
    return False


#: The one ``chmod`` long option that takes a VALUE -- the file whose mode is
#: copied.  With it present the command has no mode operand at all (its first
#: operand is a FILE), so the floor has nothing to read and says so.  GNU
#: ``getopt_long`` accepts any unambiguous abbreviation; ``--ref`` is the shortest
#: that is not also a prefix of ``--recursive``.
_REFERENCE_OPTION = "--reference"
_REFERENCE_ABBREVIATION_MIN = len("--ref")


#: A control operator the shell reads INSIDE a word, including a bare ``&``.  The
#: shared ``_CONTROL_OPERATOR_RE`` is the same class, but ``_ends_argv`` declines
#: to cut at a glued ``&`` because a redirection's ``>&`` carries one; here the
#: redirection has already been split off, so what is left of an ``&`` is bash's
#: own command boundary: ``./project&echo x`` runs ``echo`` as a NEW command.
_GLUED_BOUNDARY_RE = re.compile(r"[;&|\n]")

#: A redirection OPERATOR inside a word, the way bash lexes it: the word is
#: everything before the operator, UNLESS that prefix is a run of digits (or ``&``,
#: or ``{name}``), in which case it is the descriptor side of the redirection and
#: the word is empty.  ``+777>/dev/null`` is therefore the word ``+777`` plus a
#: redirect (measured: it sets ``777``), while ``777>/dev/null`` is fd 777
#: redirected and no word at all (measured: ``chmod: missing operand``).  ``&>``
#: and ``&>>`` are ONE operator to bash (``--recursive&>`` is the word
#: ``--recursive`` and a both-streams redirect), so they are matched before a lone
#: ``&`` can read as a command boundary; ``<&`` and ``<>`` likewise.
_REDIRECT_OPERATOR_RE = re.compile(r"&>>?|>{1,2}[&|!]?|<<<|<<-?|<[&>]?")
_DESCRIPTOR_PREFIX_RE = re.compile(r"\A(?:\d+|&|\{[A-Za-z_][A-Za-z0-9_]*\})\Z")


def _ends_in_open_operator(token: str) -> bool:
    """True if the LAST redirection operator in *token* ends the token.

    Such an operator's target is the NEXT word: ``2>``, ``&>``, ``--recursive>``,
    and a chain ``2>&1>`` alike, whose last ``>`` takes ``/dev/null`` as its
    target so it is never the mode.  A
    duplication that names its descriptor (``2>&1``) or a target in the same word
    (``>out``) is closed.
    """
    last = None
    for last in _REDIRECT_OPERATOR_RE.finditer(token):
        pass
    return last is not None and last.end() == len(token)


def _split_word(token: str) -> "tuple[str, bool, bool]":
    """``(word, ends_command, target_pending)`` for *token* as bash hands it on.

    *word* is the argv element the program receives -- the text before any
    attached redirection and before any glued control operator.  *ends_command*
    says whether a glued operator closes the command here, so the words after it
    belong to the NEXT command (``./project&echo --reference=t`` hands ``chmod`` the
    file ``./project``; ``echo`` and its option are not its).  *target_pending*
    says the token ENDS in a redirection operator, so the redirection's target is
    the NEXT word and never reaches the program: ``--recursive> /dev/null 777``
    hands ``chmod`` the argv ``--recursive 777``, not the mode ``/dev/null``.

    An attached redirection is cut first, because its own ``&`` (``>&2``,
    ``&>/dev/null``) is the redirection's, not a boundary.  A descriptor-shaped
    prefix (digits, ``&``, ``{name}``) is the redirection's too and leaves no word.
    The leading redirection shape (a word that IS a redirection, like
    ``2>/dev/null``) is the caller's case and is handled before this is asked.
    """
    redirect = _REDIRECT_OPERATOR_RE.search(token)
    tail = ""
    if redirect is not None:
        word = token[: redirect.start()]
        tail = token[redirect.end() :]
        if _DESCRIPTOR_PREFIX_RE.match(word):
            word = ""
    else:
        word = token
    boundary = _GLUED_BOUNDARY_RE.search(word)
    if boundary is not None:
        return word[: boundary.start()], True, False
    # A boundary inside the redirection's TARGET (``>out&echo x``) also ends the
    # command; the target itself is the shell's and is never a word here.
    pending = redirect is not None and _ends_in_open_operator(token)
    return word, _GLUED_BOUNDARY_RE.search(tail) is not None, pending


def _ends_in_open_redirect(token: str) -> bool:
    """True if *token* (the ``chmod`` word itself) ends in a redirection operator.

    ``chmod> /dev/null -R 777 ~`` redirects to ``/dev/null`` and hands ``chmod`` the
    argv ``-R 777 ~``: the word after the program is the redirection's target.
    """
    return _split_word(token)[2]


def _is_reference_option(option: str) -> bool:
    """True if *option* is ``--reference`` or an unambiguous abbreviation of it."""
    name = option.split("=", 1)[0]
    return len(name) >= _REFERENCE_ABBREVIATION_MIN and _REFERENCE_OPTION.startswith(name)


#: BSD ``chmod``'s ``getopt`` string is ``HLPRXfghorstuvwx`` (FreeBSD
#: ``bin/chmod/chmod.c``; macOS ships the same tool).  Lowercased, as the gate
#: hands text to this floor: ``x`` stands for ``X`` too, and ``r`` for both the
#: option ``-R`` and the mode letter ``-r``.
_BSD_FLAG_LETTERS = frozenset("hlpfv")
_BSD_MODE_LETTERS = frozenset("gostuwx")


def _bsd_mode_candidates(words: "list[str]") -> "list[str]":
    """Every word BSD ``chmod`` could take as the MODE of *words*, per its source.

    BSD ``getopt`` does not permute: it stops at the first word that is not an
    option cluster, and that word is the mode.  Inside a cluster, a mode letter
    (``-w``, ``-x``) ends option parsing: alone (``-x``) the word itself is the
    mode; as the LAST letter of a cluster (``-Rw``) getopt has already stepped past
    the word, so the NEXT word is the mode -- ``chmod -Rw 777 f`` sets ``777``
    recursively on BSD while GNU refuses ``-Rw`` as an invalid mode; in the middle
    of a cluster the cluster itself is the mode.  Any other letter, ``--`` long
    options included, is a usage error and changes nothing.  ``r`` is read both as
    ``-R`` (parsing continues) and as the mode letter, because lowercasing erased
    the difference: every candidate is evaluated and any grant refuses.
    """
    out: list[str] = []
    for index, word in enumerate(words):
        following = words[index + 1] if index + 1 < len(words) else None
        if word == "--":
            if following is not None:
                out.append(following)
            return out
        if not word.startswith("-") or word == "-":
            out.append(word)
            return out
        cluster = word[1:]
        for position, letter in enumerate(cluster):
            last = position == len(cluster) - 1
            if letter in _BSD_MODE_LETTERS or letter == "r":
                if len(cluster) == 1:
                    out.append(word)
                elif last:
                    if following is not None:
                        out.append(following)
                else:
                    out.append(word)
                if letter != "r":
                    return out
            elif letter not in _BSD_FLAG_LETTERS:
                return out
    return out


def _chmod_argv_sets_world_rwx(argv: "list[str]", target_pending: bool = False) -> bool:
    """True if *argv* (the words AFTER the ``chmod`` word) names a world-rwx mode.

    Reads the words the way the shell and then GNU ``chmod`` do.  A REDIRECTION is
    the shell's, not ``chmod``'s: ``2>/dev/null``, ``>file``, ``&>/dev/null`` and
    ``2>&1`` never reach the program's argv, and an operator with no target in its
    own word (``>``, ``2>``, ``--recursive>``) takes the NEXT word with it -- so
    ``chmod -R 2>/dev/null 777 ~`` and ``chmod -R --recursive> /dev/null 777 ~`` both
    hand ``chmod`` a ``777`` mode.  *target_pending* carries that state in from the
    ``chmod`` word itself (``chmod> /dev/null ...``).

    Each word is then brace-expanded, as bash does before ``chmod`` runs, with the
    package's shared expander; a brace word any of whose expansions would grant
    as the mode is refused outright (``{00777,0755}``), so the expander's word
    order never decides the verdict.  Of the words that remain,
    every ``--long`` and short option is an option wherever it sits (GNU
    permutes), ``--`` ends option parsing, ``--reference`` means the command
    carries NO mode operand, a ``-``-led MODE word (``-x``, ``-w,+777``,
    :func:`_is_dash_mode`) is the mode and turns every operand into a file, and
    otherwise the FIRST operand is the mode.  The verdict waits until the whole
    argv has been read, because the word that changes the reading can come AFTER
    the word it changes: ``chmod -v 777 --reference=t f`` copies ``t``'s mode, and
    ``chmod 777 -x f`` (GNU) removes ``x`` from the files ``777`` and ``f``.

    BSD ``chmod`` (macOS) does NOT permute: its ``getopt`` stops at the first word
    that is not an option cluster, and a mode letter inside a cluster stops it too
    (:func:`_bsd_mode_candidates`, read from FreeBSD's ``bin/chmod/chmod.c``).  So
    there ``chmod -v 777 --reference=t f`` and ``chmod -v 777 -x f`` set ``777``, and
    ``chmod -Rw 777 f`` sets ``777`` recursively while GNU calls ``-Rw`` invalid.
    The argv is refused when EITHER reading grants, because the gate does not know
    which ``chmod`` the line reaches.  BSD behaviour was read from source, not
    measured on a BSD host.

    The mode is then evaluated by :func:`_mode_grants_world_rwx`.  A mode word
    that is still a pathname glob after brace expansion, or a brace word past the
    expansion cap, is refused: the filesystem decides it.

    Stops at the first word that ends the argv (``;``, ``|``, ``&&``, a comment);
    a glued operator leaves the part before it to ``chmod``.
    """
    options_open = True
    reference = False
    positional: "str | None" = None
    dash_modes: list[str] = []
    # Every word the program receives, in order, for BSD's non-permuting reading.
    program_words: list[str] = []
    redirect_target_pending = target_pending
    reference_file_pending = False
    for token in argv:
        if token.startswith("#"):
            break
        # A word that IS a redirection (``2>/dev/null``, ``>``, ``&>/dev/null``) is
        # judged on the RAW word: its ``&`` is the operator's own, not a boundary.
        # It can sit between any two argv words, so it never settles a claim the
        # previous word made on the next one.
        redirect = _REDIRECT_START_RE.match(token)
        if redirect is not None:
            if _ends_in_open_operator(token):
                redirect_target_pending = True
            elif _GLUED_BOUNDARY_RE.search(token[redirect.end() :]):
                break
            continue
        head, ends_command, opens_redirect = _split_word(token)
        if redirect_target_pending:
            # This word is the previous operator's target, whatever it spells.
            redirect_target_pending = False
            words: list[str] = []
        elif not head:
            words = []
        else:
            expanded = _expand_braces(head)
            if expanded is None:
                # Past the expander's cap the words are unknown.  That matters only
                # where a word could still be the mode: the mode position is open,
                # or options are open and the word could expand to a dash-led mode.
                # A file list past the cap (``chmod 644 -- {1..300}``) is files.
                mode_open = positional is None and not dash_modes
                if mode_open or (options_open and "-" in head):
                    return True
                expanded = []
            if len(expanded) > 1 and positional is None and _brace_word_may_grant(expanded):
                return True
            words = expanded
        program_words.extend(words)
        for word in words:
            if reference_file_pending:
                reference = True
                reference_file_pending = False
            elif options_open and word == "--":
                options_open = False
            elif options_open and _is_dash_mode(word):
                dash_modes.append(word)
            elif options_open and word.startswith("-") and len(word) > 1:
                if _PATH_GLOB_RE.search(word):
                    # ``-*`` can expand to a file named ``-x,+777``: unreadable.
                    return True
                if word.startswith("--") and _is_reference_option(word):
                    if "=" in word:
                        reference = True
                    else:
                        reference_file_pending = True
            elif positional is None:
                positional = word
        if opens_redirect:
            redirect_target_pending = True
        if ends_command or _ends_argv(token):
            break
    gnu_mode = None if reference else (",".join(dash_modes) if dash_modes else positional)
    for mode in [gnu_mode, *_bsd_mode_candidates(program_words)]:
        if mode is None:
            continue
        if _PATH_GLOB_RE.search(mode):
            return True
        if _mode_grants_world_rwx(mode):
            return True
    return False


def _names_chmod_invocation(token: str, program: str, at_program: bool = False) -> bool:
    """True if *token* is the ``chmod`` that RUNS in its command, not a mention.

    *program* is what ``_argv_programs`` attributes the word to, and *at_program*
    says the word sits in program position.  Three readings:

    * The word is in program position, or an
      argument of a program that EXECUTES its arguments -- ``sudo``, ``env``,
      ``nice``, ``time``, ``xargs``, ``find -exec``, ``busybox``, ``ssh``, a
      ``docker exec``.  No such program is a data consumer, so the test is
      membership in the shared ``_DATA_CONSUMER_PROGRAMS`` vocabulary, and a
      wrapper nobody enumerated is read as executing by default (fail closed).
    * An argument of a data consumer is DATA to that program, and the words after
      it are that program's, not ``chmod``'s: ``grep -rn chmod src/``,
      ``ls -l /bin/chmod``, ``rg -n chmod *``, ``sed -i 's/chmod -R 777/…/' f``.
      This is the WHOLE consumer set, not the narrower set the inert-mention
      narrowing accepts: ``rg``, ``sed``, ``awk`` and ``echo`` are excluded there
      because they can EMIT or SPAWN, which matters for exonerating a row the regex
      already matched -- but none of them hands its argv to ``chmod``, so reading
      ``rg``'s bare ``*`` as ``chmod``'s mode would refuse an ordinary search.  The
      regex row keeps catching the literal under them (``echo chmod 777 ~``) on its
      own, and the mention narrowing keeps deciding whether that is inert.
    * ...unless a control operator is GLUED inside the word (``f|chmod``):
      ``_argv_programs`` opens a new command only between whole tokens, so it
      attributes the word to the consumer BEFORE the operator while the shell
      runs what comes after it.  ``_program_basename`` takes the trailing
      segment, so the word names ``chmod`` as a PROGRAM, and it is read as one.
    """
    if _program_basename(token) != _CHMOD_PROGRAM:
        return False
    if at_program or program not in _DATA_CONSUMER_PROGRAMS:
        # ``chmod`` is itself in the consumer vocabulary (its own arguments are
        # paths), so program POSITION is tested first: the word that IS the
        # program is the invocation, and a later ``chmod`` word owned by a chmod
        # is that chmod's file operand (``chmod 755 -- chmod 0777`` sets 755 on
        # two files).
        return True
    return _CONTROL_OPERATOR_RE.search(_REDIRECT_AMP_RE.sub("", token)) is not None


def _frame_sets_world_rwx(tokens: "list[str]") -> bool:
    """True if one command in this frame's *tokens* is a world-rwx ``chmod``."""
    programs = _argv_programs(tokens)
    expect_program = True
    for index, token in enumerate(tokens):
        at_program = expect_program and _ASSIGNMENT_RE.match(token) is None
        if _ends_argv(token):
            expect_program = True
        elif at_program:
            expect_program = False
        if not _names_chmod_invocation(token, programs[index], at_program):
            continue
        if _chmod_argv_sets_world_rwx(tokens[index + 1 :], _ends_in_open_redirect(token)):
            return True
    return False


# ── Heredoc bodies that a program only READS ──
# The shared walk folds a heredoc body into its frame as more commands (a newline
# is a ``;``), because it cannot tell ``bash <<EOF`` (runs the body) from ``cat
# <<EOF`` (prints it).  Reading the body is the fail-closed direction, but it
# refused every commit message, spec bullet and test fixture that MENTIONS a flag
# spelling of the change -- ``git commit -F - <<'EOF'``, ``cat >> notes.md <<'EOF'``,
# ``tee fixture.txt <<'EOF'``, ``git commit -m "$(cat <<'EOF' ...)"`` -- none of which
# runs it.  So this floor, and only this floor, reads such a body as the data it
# is, under two conditions that are each checked on structure, never on a
# spelling:
#
# * the body's consumer (the program of the command carrying the ``<<``) is one
#   whose stdin is DATA: the shared data-consumer vocabulary minus the programs
#   that can run their stdin as a script (``sed -f -``, ``awk -f -``, the pagers'
#   ``!command``), plus ``git`` and ``gh``, which read a message or a body from
#   stdin and never execute it;
# * nothing in the line can run PRINTED text: no shell, ``eval``, ``source``,
#   ``.`` or ``xargs`` anywhere in any frame, and no expansion in a position
#   where its OUTPUT would be a command -- program position, or an argument of a
#   program outside that data vocabulary (``sudo "$(cat <<EOF ...)"``).  A
#   substitution as an argument of ``git``/``gh``/``cat`` is a message or a path.
#
# Everything else is read exactly as before.  Substitutions INSIDE a body are
# separate frames of the walk and are judged on their own either way.
_HEREDOC_DATA_PROGRAMS = (_DATA_CONSUMER_PROGRAMS - {"sed", "awk", "less", "more"}) | {"git", "gh"}
_PRINTED_TEXT_RUNNERS = _NESTED_SHELL_PROGRAMS | _NESTED_SHELL_VERBS | {"xargs"}

#: A ``<<`` that opens a heredoc (not a ``<<<`` here-string), anywhere in a word.
_HEREDOC_OPEN_RE = re.compile(r"(?<!<)<<(?!<)")
_TAG_QUOTES = "'\"\\"


def _heredoc_tag(tokens: "list[str]", index: int) -> "tuple[str, bool] | None":
    """``(tag, strips_tabs)`` for a heredoc opened in ``tokens[index]``, else None."""
    token = tokens[index]
    found = _HEREDOC_OPEN_RE.search(token)
    if found is None:
        return None
    marker = _heredoc_marker(token[found.start() :])
    if marker is None:
        return None
    strips_tabs = token[found.start() :].startswith("<<-")
    if marker == "":
        if index + 1 >= len(tokens):
            return None
        marker = tokens[index + 1]
    tag = re.split(r"[;&|]", marker)[0]
    tag = "".join(ch for ch in tag if ch not in _TAG_QUOTES)
    return (tag, strips_tabs) if tag else None


def _active_heredoc_openers(source: str) -> int:
    """How many UNQUOTED ``<<`` heredoc operators *source* holds.

    A ``<<`` inside quotes is data (``git commit -m "use << here"``), and the
    tokens the floor reads have had their quotes removed, so the count is taken on
    the raw text through the shell's one quote/escape state machine, so a quoted
    ``<<`` never hides the lines after it.
    """
    chars = list(_iter_shell_chars(source))
    count = 0
    for index, step in enumerate(chars):
        if not (step.active and step.char == "<"):
            continue
        prev_lt = index > 0 and chars[index - 1].active and chars[index - 1].char == "<"
        nxt = chars[index + 1] if index + 1 < len(chars) else None
        after = chars[index + 2] if index + 2 < len(chars) else None
        if prev_lt or nxt is None or not (nxt.active and nxt.char == "<"):
            continue
        if after is not None and after.active and after.char == "<":
            continue  # ``<<<`` is a here-string, not a heredoc
        count += 1
    return count


#: ``git`` and ``gh`` read a heredoc as a message or a body only through a BUILTIN
#: subcommand: a ``git`` alias can be ``!sh`` (``git -c alias.x='!sh' x <<EOF``
#: runs the body), and neither tool lets an
#: alias shadow a builtin (git: "aliases that would hide existing git commands are
#: ignored"), so a ``-c`` that defines one cannot reach a builtin.  ``-c`` and
#: ``-C`` look the same in the gate's lowercased text; both take one value.
_HEREDOC_DATA_SUBCOMMANDS = {
    "git": frozenset({"commit", "tag", "notes", "hash-object"}),
    "gh": frozenset({"pr", "issue", "release", "api", "gist"}),
}
_GIT_OPTIONS_WITH_VALUE = frozenset({"-c", "--config-env", "--git-dir", "--work-tree"})


def _reads_heredoc_as_data(words: "list[str]") -> bool:
    """True if the command *words* (program first) only READS its stdin as data."""
    program = _program_basename(words[0])
    if program not in _HEREDOC_DATA_PROGRAMS:
        return False
    builtins = _HEREDOC_DATA_SUBCOMMANDS.get(program)
    if builtins is None:
        return True
    rest = iter(words[1:])
    for word in rest:
        if word in _GIT_OPTIONS_WITH_VALUE:
            next(rest, None)
            continue
        if word.startswith("-"):
            continue
        return word in builtins
    return False


def _without_data_heredoc_body(source: str, tokens: "list[str]") -> "list[str] | None":
    """*tokens* re-read with a data consumer's heredoc body removed, else None.

    Only the simple shape is handled: exactly one UNQUOTED heredoc operator in the
    frame (:func:`_active_heredoc_openers`), in one token, opened on one line of
    *source*, and consumed by a command that reads stdin as data
    (:func:`_reads_heredoc_as_data`).  The body runs from the line after it to the
    first line that is exactly the tag (``<<-`` strips leading tabs first), as bash
    ends it.  A body with no closing tag line is kept.  Every other shape keeps its
    body, the fail-closed reading.
    """
    openers = [i for i, token in enumerate(tokens) if _HEREDOC_OPEN_RE.search(token)]
    if len(openers) != 1 or len(_HEREDOC_OPEN_RE.findall(source)) != 1:
        return None
    if _active_heredoc_openers(source) != 1:
        return None
    index = openers[0]
    start = index
    while start > 0 and not _ends_argv(tokens[start - 1]):
        start -= 1
    while start < index and _ASSIGNMENT_RE.match(tokens[start]):
        start += 1
    if not _reads_heredoc_as_data(tokens[start:index] or [tokens[index]]):
        return None
    tag_info = _heredoc_tag(tokens, index)
    if tag_info is None:
        return None
    tag, strips_tabs = tag_info
    lines = source.split("\n")
    opener_line = next(i for i, line in enumerate(lines) if _HEREDOC_OPEN_RE.search(line))
    for i in range(opener_line + 1, len(lines)):
        if (lines[i].lstrip("\t") if strips_tabs else lines[i]) == tag:
            return _self_tokens("\n".join(lines[: opener_line + 1] + lines[i:]))
    return None


#: A word that opens with an expansion once the quotes and parentheses the shell
#: strips first are off: its OUTPUT is what the word becomes.
_EXPANSION_LEAD_RE = re.compile(r"\A[\"'(]*[$`]")

#: A leading ``NAME=value`` word, which precedes the program instead of being it.
_ASSIGNMENT_RE = re.compile(r"\A[a-z_][a-z0-9_]*=")


def _can_run_printed_text(frames: "list[list[str]]") -> bool:
    """True if any frame could execute text another command PRINTS.

    A word counts when it sits where its text becomes a command: program position
    (past any leading assignments), or an argument of a program outside the data
    vocabulary (``sudo``, ``env``, ``find -exec``).  Such a word is a shell, an
    evaluator or ``xargs`` by name, or an expansion whose output is the command.
    """
    for tokens in frames:
        programs = _argv_programs(tokens)
        expect_program = True
        for index, token in enumerate(tokens):
            at_program = expect_program and _ASSIGNMENT_RE.match(token) is None
            if at_program or programs[index] not in _HEREDOC_DATA_PROGRAMS:
                if _program_basename(token) in _PRINTED_TEXT_RUNNERS:
                    return True
                if _EXPANSION_LEAD_RE.match(token):
                    return True
            if _ends_argv(token):
                expect_program = True
            elif at_program:
                expect_program = False
    return False


#: Characters without which no word can de-quote or expand into a spelling it
#: does not literally contain.  The frame walk resolves quoting, escapes, ANSI-C
#: literals, empty-quote splices, same-line assignments and printf escapes in a
#: nested payload -- each needs one of these in the raw text -- so a command
#: carrying none of them and not the literal word can never reach the predicate.
#: The pre-filter is a NECESSARY condition for a hit, never the verdict: a false
#: positive here only pays for the walk the self-protection floors pay anyway.
_GLUE_CHARS_RE = re.compile(r"['\"\\$`]")


def _chmod_floor_can_fire(text_lower: str) -> bool:
    """Cheap necessary condition for :func:`_is_chmod_world_rwx` to return True."""
    return _CHMOD_PROGRAM in text_lower or _GLUE_CHARS_RE.search(text_lower) is not None


def _is_chmod_world_rwx(text_lower: str) -> bool:
    """True if *text_lower* runs ``chmod`` with a world-rwx numeric mode.

    Judged on the argv of the command itself and of every nested shell payload
    (``bash -c``, ``eval``, ``$( )``, heredocs), so a wrapper buys nothing.  A
    heredoc body that a data consumer only reads is set aside first, when nothing
    in the line can run printed text (see ``_HEREDOC_DATA_PROGRAMS``).  The
    contract is the floor's: callers pass already-lowercased text, as
    ``is_denied`` does.

    Runs inside the PreToolUse gate, which must return a DECISION and never
    raise, so an exception from the walk is a False here: the regex row this
    floor is a union with is still in the tier, and that is what keeps a reader
    hiccup from being a bypass of the literal spelling.
    """
    if not _chmod_floor_can_fire(text_lower):
        return False
    try:
        frames = _shell_payload_walk(text_lower)
    except Exception:
        return False
    views: list[list[str]] = []
    for source, tokens in frames:
        try:
            stripped = _without_data_heredoc_body(source, tokens)
        except Exception:
            stripped = None
        views.append(tokens if stripped is None else stripped)
    try:
        if _can_run_printed_text(views):
            views = [tokens for _source, tokens in frames]
    except Exception:
        views = [tokens for _source, tokens in frames]
    for tokens in views:
        try:
            if _frame_sets_world_rwx(tokens):
                return True
        except Exception:
            continue
    return False
