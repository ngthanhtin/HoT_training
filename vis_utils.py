import re
from html import escape

# ------------------------------------------------------------------
# 1. colour palette
# ------------------------------------------------------------------
char_2_color_map = {
    'think': "#808080",
    'fact1': "#FF5733", 'fact2': "#33FF57", 'fact3': "#3357FF",
    'fact4': "#FF33A1", 'fact5': "#FFA533", 'fact6': "#33FFF3",
    'fact7': "#FFDD33", 'fact8': "#8D33FF", 'fact9': "#33FF8D",
    'fact10': "#FF335E", 'fact11': "#3378FF", 'fact12': "#FFB833",
    'fact13': "#FF33F5", 'fact14': "#75FF33", 'fact15': "#33C4FF",
    'fact16': "#FF8633", 'fact17': "#C433FF", 'fact18': "#33FFB5",
    'fact19': "#FF336B",
    # lighter “derive” shades …
    **{f"derive{i}": c.replace('#', '#85', 1) for i, c in enumerate(
        ["#FF5733", "#33FF57", "#3357FF", "#FF33A1", "#FFA533",
         "#33FFF3", "#FFDD33", "#8D33FF", "#33FF8D", "#FF335E",
         "#3378FF", "#FFB833", "#FF33F5", "#75FF33", "#33C4FF",
         "#FF8633", "#C433FF", "#33FFB5", "#FF336B"], start=1)}
}

# small CSS helper
def _span(tag_name, inner):
    color = char_2_color_map.get(tag_name, "#cccccc")
    return (f'<span style="background:{color};padding:1px 3px;'
            f'border-radius:3px;">{escape(inner)}</span>')

# ------------------------------------------------------------------
# 2. generic highlighter
# ------------------------------------------------------------------
_tag_re = re.compile(r"</?(think|fact\d+|derive\d+)(?:\s[^>]*)?>", re.I)

def _highlight_no_coords(text: str) -> str:
    out, pos, stk = [], 0, []
    for m in _tag_re.finditer(text):
        out.append(escape(text[pos:m.start()]))
        tag = m.group(1).lower()
        if text[m.start()+1] != '/':                 # open
            out.append(f'§{tag}§')                  # placeholder
            stk.append(tag)
        else:                                       # close
            if stk and stk[-1] == tag:
                inner = ''.join(out[out.index(f'§{tag}§')+1:])
                # replace opening placeholder + inner with coloured span
                start = out.index(f'§{tag}§')
                out[start:] = [_span(tag, inner)]
                stk.pop()
        pos = m.end()
    out.append(escape(text[pos:]))
    return ''.join(out)

    
def _highlight(text: str) -> str:
    """
    Replace the xml-ish tags with coloured <span> markup; keep raw text outside.
    """
    out = []
    pos = 0
    stack = []      # track nested tag order
    for m in _tag_re.finditer(text):
        out.append(escape(text[pos:m.start()]))      # plain chunk
        tag = m.group(1).lower()
        if text[m.start()+1] != '/':                # opening tag
            stack.append((tag, len(out)))
        else:                                       # closing tag
            if stack and stack[-1][0] == tag:
                tag_open, idx = stack.pop()
                # inner text = everything appended since opening
                inner = ''.join(out[idx:])
                out[idx:] = [_span(tag_open, inner)]
        pos = m.end()
    out.append(escape(text[pos:]))                  # tail
    # close any un-closed tags naively
    while stack:
        tag_open, idx = stack.pop()
        inner = ''.join(out[idx:])
        out[idx:] = [_span(tag_open, inner)]
    return ''.join(out)

# Function to extract parts of the text based on headers
def extract_parts(text):
    # Find the "Reformatted Question" and "Answer" sections
    question_match = re.search(r"Reformatted Question:(.*?)\nAnswer:", text, re.S)
    answer_match = re.search(r"Answer:(.*)", text, re.S)

    # Extracting text for each part
    if question_match:
        question_text = question_match.group(1).strip()
    else:
        question_match = re.search(r"Reformatted Question:(.*?)Answer:", text, re.S)
        answer_match = re.search(r"Answer:(.*)", text, re.S)
        
        if question_match:
            question_text = question_match.group(1).strip()
        else:
            question_text = "Question not found"
    
    if answer_match:
        answer_text = answer_match.group(1).strip()
    else:
        answer_text = "Answer not found"

    return question_text, answer_text
# ------------------------------------------------------------------
# 3-A. localisation version – expects x1/x2 attrs but we ignore them
# ------------------------------------------------------------------
def visualize_hot_localization(response: str) -> str:
    """Return HTML that highlights HoT localisation spans in question & answer."""
    q_m = re.search(r"<question>(.*?)</question>", response, re.S|re.I)
    a_m = re.search(r"<answer>(.*?)</answer>",   response, re.S|re.I)
    if not (q_m and a_m):
        raise ValueError("Response must contain <question> … </question> and <answer> … </answer>")

    question_txt = q_m.group(1).strip()
    answer_txt   = a_m.group(1).strip()

    # --------------------------------------------------------
    # 1. collect (tag, x1, x2) triples from *answer*
    # --------------------------------------------------------
    coord_pat = re.compile(r"<fact(\d+)\s+x1=(\d+)\s+x2=(\d+)>", re.I)
    spans = [(f"fact{m.group(1)}", int(m.group(2)), int(m.group(3)))
             for m in coord_pat.finditer(answer_txt)]

    # strip the coordinates so answer now has simple <factN> tags
    answer_clean = re.sub(r"\s+x1=\d+\s+x2=\d+", "", answer_txt)
    answer_high  = _highlight_no_coords(answer_clean)

    # --------------------------------------------------------
    # 2. colour the matching word ranges in the *question*
    # --------------------------------------------------------
    # keep spaces so we preserve original formatting
    toks = re.findall(r'\S+|\s+', question_txt)
    word_pos = [i for i, t in enumerate(toks) if not t.isspace()]

    open_at  = {}    # token-idx → opening <span>
    close_at = {}    # token-idx → closing </span>

    for tag, x1, x2 in spans:
        if x1 >= len(word_pos) or x2 >= len(word_pos):          # out of range → skip
            continue
        s_idx, e_idx = word_pos[x1], word_pos[x2]
        color = char_2_color_map.get(tag, "#cccccc")
        open_at.setdefault(s_idx, []).append(
            f'<span style="background:{color};padding:1px 3px;border-radius:3px;">')
        close_at.setdefault(e_idx, []).append('</span>')

    q_out = []
    for i, tok in enumerate(toks):
        if i in open_at:
            q_out.extend(open_at[i])
        q_out.append(escape(tok))
        if i in close_at:
            q_out.extend(close_at[i])

    question_high = ''.join(q_out)

    # --------------------------------------------------------
    # 3. final HTML blob
    # --------------------------------------------------------
    return f"""
    <div style="font-family:system-ui;line-height:1.5">
      <h3>Question</h3>
      <p>{question_high}</p>

      <h3>Answer</h3>
      <p>{answer_high}</p>
    </div>
    """

# ------------------------------------------------------------------
# 3-B. plain version – the “Reformatted Question / Answer” style
# ------------------------------------------------------------------
def visualize_hot(response: str) -> str:
    """
    Converts a HoT response that contains 'Reformatted Question:' and
    'Answer:' sections into coloured HTML.
    """
    ref_question, answer = extract_parts(response)
    ref_q  = _highlight(ref_question.strip())
    ref_ans = _highlight(answer.strip())

    html = f"""
    <div style="font-family:system-ui;line-height:1.45">
      <h3>Reformatted Question</h3>
      <p>{ref_q}</p>
      <h3>Answer</h3>
      <p>{ref_ans}</p>
    </div>
    """
    return html
