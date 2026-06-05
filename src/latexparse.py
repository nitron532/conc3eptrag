import json
import re
import sys
from typing import Optional


# ---------------------------------------------------------------------------
# LaTeX cleaning helpers
# ---------------------------------------------------------------------------

def remove_verbatim_blocks(text: str) -> tuple[str, list[str]]:
    """
    Extract verbatim blocks before general cleaning so their content is
    preserved exactly (no backslash-stripping inside code).
    Returns (text_with_placeholders, [verbatim_contents])
    """
    verbatim_blocks = []
    placeholder_base = "\x00VERBATIM{}\x00"

    def replace_verbatim(m):
        verbatim_blocks.append(m.group(1))
        return placeholder_base.format(len(verbatim_blocks) - 1)

    text = re.sub(
        r'\\begin\{verbatim\}(.*?)\\end\{verbatim\}',
        replace_verbatim,
        text,
        flags=re.DOTALL
    )
    return text, verbatim_blocks


def restore_verbatim_blocks(text: str, verbatim_blocks: list[str]) -> str:
    def restore(m):
        idx = int(m.group(1))
        return verbatim_blocks[idx].strip('\n')
    return re.sub(r'\x00VERBATIM(\d+)\x00', restore, text)


def clean_latex(text: str) -> str:
    """
    Apply LaTeX cleaning rules to produce plain readable text.
    Verbatim blocks should already be extracted before calling this.
    """

    # --- tcolorbox: remove \begin{tcolorbox}[...] entirely (options can span lines)
    text = re.sub(r'\\begin\{tcolorbox\}(?:\[[^\]]*\])?(?:\{[^}]*\})*', '', text, flags=re.DOTALL)
    text = re.sub(r'\\end\{tcolorbox\}', '', text)

    # --- minipage: remove \begin{minipage}{...} and \end{minipage}, keep content
    text = re.sub(r'\\begin\{minipage\}(?:\[[^\]]*\])?\{[^}]*\}', '', text)
    text = re.sub(r'\\end\{minipage\}', '', text)

    # --- vspace / hspace: remove entirely including optional * and argument
    text = re.sub(r'\\[vh]space\*?\{[^}]*\}', '', text)

    # --- color commands: remove \color{x}, keep inner text of \textcolor{x}{...}
    text = re.sub(r'\\textcolor\{[^}]*\}\{(.*?)\}', r'\1', text, flags=re.DOTALL)
    text = re.sub(r'\\color\{[^}]*\}', '', text)

    # --- mbox: keep inner content
    text = re.sub(r'\\mbox\{(.*?)\}', r'\1', text, flags=re.DOTALL)

    # --- LaTeX line comments: % to end of line (but not \% escaped percent)
    text = re.sub(r'(?<!\\)%[^\n]*', '', text)

    # --- textbf / texttt / textit / emph / text: keep inner content
    for cmd in ('textbf', 'texttt', 'textit', 'emph', 'text', 'textrm', 'textsc', 'underline'):
        text = re.sub(rf'\\{cmd}\{{(.*?)\}}', r'\1', text, flags=re.DOTALL)

    # --- special symbols: keep the symbol, remove the backslash
    text = re.sub(r'\\([@%\$&_#])', r'\1', text)

    # --- item / itemize / enumerate / description environments
    text = re.sub(r'\\begin\{(?:itemize|enumerate|description)\}', '', text)
    text = re.sub(r'\\end\{(?:itemize|enumerate|description)\}', '', text)
    text = re.sub(r'\\item\b', '-', text)  # replace \item with a dash

    # --- common formatting / structural commands to drop entirely
    drop_commands_no_arg = [
        'noindent', 'newline', 'par', 'centering', 'raggedright', 'raggedleft',
        'small', 'large', 'Large', 'huge', 'Huge', 'normalsize', 'footnotesize',
        'vfill', 'hfill', 'clearpage', 'newpage', 'pagebreak',
        'medskip', 'bigskip', 'smallskip',
    ]
    for cmd in drop_commands_no_arg:
        text = re.sub(rf'\\{cmd}\b\s*', '', text)


    # --- commands with one mandatory argument to drop entirely (incl. argument)
    drop_commands_one_arg = [
        'label', 'ref', 'pageref', 'cite', 'footnote',
        'includegraphics', 'caption', 'hyperref', 'geometry',
    ]
    for cmd in drop_commands_one_arg:
        text = re.sub(rf'\\{cmd}(?:\[[^\]]*\])?\{{[^}}]*\}}', '', text)

    # --- two-arg commands: setlength/addtolength take \cmd and value args
    for cmd in ('setlength', 'addtolength'):
        text = re.sub(rf'\\{cmd}\{{[^}}]*\}}\{{[^}}]*\}}', '', text)

    # --- remaining \begin{...} / \end{...} wrappers (keep content, remove tags)
    text = re.sub(r'\\(?:begin|end)\{[^}]*\}', '', text)

    # --- any remaining \command (no braces) — remove the command word only
    text = re.sub(r'\\[a-zA-Z]+\*?\b', '', text)

    # --- stray braces left over
    text = re.sub(r'[{}]', '', text)

    # --- LaTeX line breaks \\ → newline
    text = re.sub(r'\\\\+', '\n', text)

    # --- collapse excessive whitespace / blank lines
    text = re.sub(r'\n{3,}', '\n\n', text)
    text = re.sub(r'[ \t]+', ' ', text)
    text = text.strip()

    return text


# ---------------------------------------------------------------------------
# Question / Answer extraction
# ---------------------------------------------------------------------------

# Environments that constitute the answer
ANSWER_ENVS = re.compile(
    r'\\begin\{(Solution|solution|AnswerArea|answerarea|Answer|answer)\}(.*?)\\end\{\1\}',
    re.DOTALL | re.IGNORECASE
)

# Environments to strip from question entirely (answer boxes, blank fill areas)
STRIP_FROM_QUESTION_ENVS = re.compile(
    r'\\begin\{(?:Question|tcolorbox|AnswerArea|answerarea|Answer|answer|Solution|solution)[^}]*\}.*?\\end\{[^}]*\}',
    re.DOTALL | re.IGNORECASE
)

# \newcommand definitions to strip entirely before any other processing
NEWCOMMAND_RE = re.compile(
    r'\\(?:newcommand|renewcommand|providecommand)\{[^}]*\}(?:\[\d+\])?\{',
    re.DOTALL
)


def extract_balanced_braces(text: str, start: int) -> Optional[str]:
    """
    Given text and the index of an opening '{', walk forward tracking
    brace depth and return the content inside the outermost braces.
    Skips over LaTeX line comments (% to end of line) and escaped braces (\{ \}).
    Returns None if braces are unbalanced.
    """
    assert text[start] == '{'
    depth = 0
    i = start
    while i < len(text):
        ch = text[i]
        # skip escaped characters so \{ and \} don't affect depth
        if ch == '\\' and i + 1 < len(text):
            i += 2
            continue
        # skip LaTeX line comments: % to end of line
        if ch == '%':
            while i < len(text) and text[i] != '\n':
                i += 1
            continue
        if ch == '{':
            depth += 1
        elif ch == '}':
            depth -= 1
            if depth == 0:
                return text[start + 1:i]
        i += 1
    return None  # unbalanced


def strip_newcommand_definitions(text: str, _macros: dict | None = None) -> tuple[str, dict[str, str]]:
    """
    Remove \\newcommand{\\name}{body} definitions from text and return a dict
    mapping command name → body so we can expand them in the question text.
    Only simple zero-argument commands are expanded; parametric ones are dropped.
    Uses recursion to handle multiple definitions; _macros accumulates across calls.
    """
    macros: dict[str, str] = _macros if _macros is not None else {}
    result = text

    for m in NEWCOMMAND_RE.finditer(text):
        body_start = m.end() - 1  # points to the opening '{' of the body
        body = extract_balanced_braces(text, body_start)
        if body is None:
            continue

        name_match = re.search(r'\\(?:newcommand|renewcommand|providecommand)\{(\\[a-zA-Z]+)\}', m.group(0))
        if name_match:
            cmd_name = name_match.group(1)
            if '[' not in m.group(0):  # zero-arg macros only
                macros[cmd_name] = body

        full_end = body_start + len(body) + 2  # +2 for outer braces
        result = result[:m.start()] + result[full_end:]
        # restart on modified string, passing accumulated macros dict
        return strip_newcommand_definitions(result, macros)

    return result, macros


def expand_macros(text: str, macros: dict[str, str]) -> str:
    """Expand zero-argument \\newcommand macros in text."""
    for cmd_name, body in macros.items():
        # escape the command name for use in regex
        escaped = re.escape(cmd_name)
        text = re.sub(rf'{escaped}(?![a-zA-Z])', lambda m, b=body: b, text)
    return text


def extract_solution_command(content: str) -> tuple[Optional[str], str]:
    """
    Handle the \\solution{...} command-style syntax (as opposed to
    \\begin{Solution}...\\end{Solution} environment syntax).
    Uses balanced brace matching to correctly handle nested braces.
    Returns (solution_content_or_None, content_with_solution_removed).
    """
    pattern = re.compile(r'\\solution\s*\{', re.IGNORECASE)
    m = pattern.search(content)
    if not m:
        return None, content

    brace_start = m.end() - 1  # index of the opening '{'
    body = extract_balanced_braces(content, brace_start)
    if body is None:
        return None, content

    full_end = brace_start + len(body) + 2
    content_without = content[:m.start()] + content[full_end:]
    return body, content_without


def extract_question_and_answer(content: str) -> tuple[str, Optional[str]]:
    """
    Split raw LaTeX content into (question_text, answer_text).
    Both are returned as cleaned plain text.

    Extraction order (each method only fires if previous found nothing):
      1. \\begin{Solution} / \\begin{AnswerArea} environment
      2. \\solution{...} command with balanced brace matching
    """

    # 1. Strip \newcommand definitions — do NOT expand globally yet.
    #    Expansion happens per-section so macro bodies survive answer extraction.
    content, macros = strip_newcommand_definitions(content)

    # 2. Pull verbatim blocks out so they survive cleaning intact
    content, verbatim_blocks = remove_verbatim_blocks(content)

    # ------------------------------------------------------------------ #
    # ANSWER EXTRACTION — methods tried in order, stop at first success   #
    # ------------------------------------------------------------------ #

    raw_answer_parts: list[str] = []

    # Method 1: \begin{Solution} / \begin{AnswerArea} environment
    for m in ANSWER_ENVS.finditer(content):
        raw_answer_parts.append(m.group(2))

    # Method 2: \solution{...} command style
    solution_cmd_body: Optional[str] = None
    if not raw_answer_parts:
        solution_cmd_body, content = extract_solution_command(content)
        if solution_cmd_body:
            raw_answer_parts.append(solution_cmd_body)
    else:
        # env-style answer found — still remove \solution{} from question text
        _, content = extract_solution_command(content)

    # ------------------------------------------------------------------ #
    # QUESTION TEXT                                                        #
    # ------------------------------------------------------------------ #

    question_raw = STRIP_FROM_QUESTION_ENVS.sub('', content)
    question_raw = re.sub(r'\\solution\\s*\\{[^}]*\\}', '', question_raw, flags=re.IGNORECASE)
    question_raw = expand_macros(question_raw, macros)
    question_text = clean_latex(question_raw)
    question_text = restore_verbatim_blocks(question_text, verbatim_blocks)

    # ------------------------------------------------------------------ #
    # ANSWER TEXT — expand macros before cleaning so references resolve   #
    # ------------------------------------------------------------------ #

    if raw_answer_parts:
        answer_raw = '\n'.join(raw_answer_parts)
        answer_raw = expand_macros(answer_raw, macros)
        answer_text: Optional[str] = clean_latex(answer_raw)
        answer_text = restore_verbatim_blocks(answer_text, verbatim_blocks)
        if not answer_text or not answer_text.strip():
            answer_text = None
    else:
        answer_text = None

    question_text = re.sub(' +', ' ',   question_text)
    if answer_text:
        answer_text = re.sub(' +', ' ',   answer_text)

    return question_text.strip(), answer_text.strip() if answer_text else None


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def parse_question_bank(input_path: str, output_path: str) -> None:
    with open(input_path, 'r', encoding='utf-8') as f:
        raw = f.read()

    # JSON files may use single quotes (Python dict reprs) — normalise if needed
    try:
        questions = json.loads(raw)
    except json.JSONDecodeError:
        # try evaluating as Python literal (single-quote dicts)
        import ast
        questions = ast.literal_eval(raw)

    results = []
    for q in questions:
        content = q.get('content', '')
        question_text, answer_text = extract_question_and_answer(content)
        results.append({
            'id': q.get('id'),
            'course': q.get('course'),
            'type': q.get('type'),
            'diff': q.get('diff'),
            'tags': q.get('tags', []),
            'question': question_text,
            'answer': answer_text,
        })

    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    print(f"Parsed {len(results)} questions → {output_path}")


# ---------------------------------------------------------------------------
# Quick smoke-test on the example question
# ---------------------------------------------------------------------------

EXAMPLE = r"""
\item
Suppose you have a LinkedList and Node classes defined for singly-linked-lists. What is the bug with this code?
\begin{verbatim}
LinkedList ll;
ll.head = new Node;
ll.head->data = 3;
ll.head->next = new Node;
ll.head->next->data = 4;
ll.head->next->next = new Node;
ll.head->next->next->data = 5;
ll.head->next->next->next = nullptr;

delete ll.head->next->next;

// find tail
Node* it = ll.head;
while (it->next) it = it->next;
\end{verbatim}
\begin{Question}
\begin{tcolorbox}[colback=white, colframe=black, width=6.7in, arc=0mm, boxrule=1pt]
\vspace{5in}
\end{tcolorbox}
\end{Question}

\begin{Solution}
After deleting \texttt{ll.head->next->next}, it is never set to nullptr. This means that the data that was there is uninitialized, and could contain ``junk" values that could cause a segfault or incorrect behavior.
\end{Solution}
"""


EXAMPLE_2 = r"""
\newcommand{\cppasciicode}{%
\ttfamily
\#include <iostream>\\
using namespace std;\\[0.8em]
int main() \{\\
\hspace*{1.5em}int sum = 5;\\
\hspace*{1.5em}sum += '6';\\
\hspace*{1.5em}cout << sum << endl;\\
\}
}

\item Characters in C++ (and in many other programming languages) are stored as numbers according to the ASCII standard. In ASCII, the characters '0' to '9' are sequentially ordered and their numeric values are 48 to 57, respectively. The following code outputs 59. Modify one line of the code so it outputs 11 using the above hint. Explain your answer.

\begin{Question}
\noindent
\begin{minipage}[t]{0.3\textwidth}
\vspace{0pt}
\cppasciicode
\end{minipage}\hspace{0pt}%
\begin{minipage}[t]{0.6\textwidth}
\vspace{0pt}
\begin{tcolorbox}[colback=white,colframe=black,width=\linewidth,height=1.5in,arc=0mm,boxrule=1pt]
\mbox{}
\end{tcolorbox}
\end{minipage}
\end{Question}

\solution{
\noindent
\begin{minipage}[t]{0.3\textwidth}
\vspace{0pt}
{\color{black}\cppasciicode}
\end{minipage}\hspace{0pt}%
\begin{minipage}[t]{0.6\textwidth}
\vspace{0pt}
\begin{tcolorbox}[colback=white,colframe=black,width=\linewidth,height=1.5in,arc=0mm,boxrule=1pt]
{\color{red}\ttfamily
sum += '6' - '0'; \\
This change converts the character \texttt{'6'} to its numeric value 6 (because \texttt{'6' - '0' = 54 - 48 = 6}). Adding this to the original value of \texttt{sum} (which is 5), the final output becomes 11.
}
\end{tcolorbox}
\end{minipage}
}
"""


EXAMPLE_3 = r"""
\newcommand{\centssolutioncode}{%
{\ttfamily
// Takes a double value representing an amount of money in dollars and cents\\
int number_of_cents(double change) \\{\\
\\hspace*{1.5em}double cents = fmod(change, 1);\\
\\hspace*{1.5em}return (cents * 100);\\
\\}\\
}%
}

\item \textbf{[5 pts]} Use \code{fmod} to write a complete program.

\begin{Question}
\begin{tcolorbox}[colback=white, colframe=black, width=6.7in, height=6.8in, arc=0mm, boxrule=1pt]
\mbox{}
\end{tcolorbox}
\end{Question}

\solution{
\begin{tcolorbox}[colback=white, colframe=black, width=6.7in, height=6.8in, arc=0mm, boxrule=1pt]
{\color{red}\centssolutioncode}
\end{tcolorbox}
}
"""


if __name__ == '__main__':
    if len(sys.argv) == 3:
        parse_question_bank(sys.argv[1], sys.argv[2])
    else:
        print("No arguments provided — running smoke tests.\n")
        print("Usage: python parse_questions.py <input.json> <output.json>\n")

        print("=" * 60)
        print("SMOKE TEST 1: \\begin{Solution} environment style")
        print("=" * 60)
        q, a = extract_question_and_answer(EXAMPLE)
        print("QUESTION:")
        print(q)
        print("\nANSWER:")
        print(a)

        print("\n" + "=" * 60)
        print("SMOKE TEST 2: \\solution{} command style + \\newcommand expansion")
        print("=" * 60)
        q2, a2 = extract_question_and_answer(EXAMPLE_2)
        print("QUESTION:")
        print(q2)
        print("\nANSWER:")
        print(a2)

        print("\n" + "=" * 60)
        print("SMOKE TEST 3: \\solution{} with macro reference inside tcolorbox")
        print("=" * 60)
        q3, a3 = extract_question_and_answer(EXAMPLE_3)
        print("QUESTION:")
        print(q3)
        print("\nANSWER:")
        print(a3)