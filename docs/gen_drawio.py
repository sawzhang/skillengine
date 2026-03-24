"""Generate skillengine-execution-flow.drawio using COMPRESSED format (deflate+base64+urlencode).
This matches the format used by draw.io's native save, proven to work in test-minimal.drawio.
Uses numeric IDs to match draw.io conventions.
"""
import base64, zlib, urllib.parse
from xml.sax.saxutils import escape

SEP = "\u2500" * 14
SEP2 = "\u2500" * 18
SEP3 = "\u2500" * 22

lines = []
_next_id = [2]  # start at 2 (0 and 1 are reserved)

def w(s):
    lines.append(s)

def auto_id():
    """Generate next numeric ID."""
    val = str(_next_id[0])
    _next_id[0] += 1
    return val

def v(text):
    result = escape(text, {'"': "&quot;"})
    result = result.replace("\n", "&#xa;")
    return result

def box(val, style, x, y, width, height):
    id = auto_id()
    w(f'    <mxCell id="{id}" value="{v(val)}" style="{style}" vertex="1" parent="1">')
    w(f'      <mxGeometry x="{x}" y="{y}" width="{width}" height="{height}" as="geometry"/>')
    w(f'    </mxCell>')
    return id

def edge(style, src, tgt, pts=None, label=None, label_style=None):
    id = auto_id()
    if pts:
        w(f'    <mxCell id="{id}" style="{style}" edge="1" source="{src}" target="{tgt}" parent="1">')
        w(f'      <mxGeometry relative="1" as="geometry">')
        w(f'        <Array as="points">')
        for px, py in pts:
            w(f'          <mxPoint x="{px}" y="{py}"/>')
        w(f'        </Array>')
        w(f'      </mxGeometry>')
        w(f'    </mxCell>')
    else:
        w(f'    <mxCell id="{id}" style="{style}" edge="1" source="{src}" target="{tgt}" parent="1">')
        w(f'      <mxGeometry relative="1" as="geometry"/>')
        w(f'    </mxCell>')
    if label:
        ls = label_style or "edgeLabel;html=1;fontSize=9;fontStyle=2;"
        lid = auto_id()
        w(f'    <mxCell id="{lid}" value="{v(label)}" style="{ls}" vertex="1" connectable="0" parent="{id}">')
        w(f'      <mxGeometry relative="1" as="geometry"><mxPoint as="offset"/></mxGeometry>')
        w(f'    </mxCell>')
    return id

def compress_xml(xml_str):
    """Compress XML: deflate -> base64. Raw base64 output (no URI encoding)."""
    xml_bytes = xml_str.encode("utf-8")
    compressor = zlib.compressobj(zlib.Z_DEFAULT_COMPRESSION, zlib.DEFLATED, -15)
    compressed = compressor.compress(xml_bytes) + compressor.flush()
    return base64.b64encode(compressed).decode("ascii")

def main():
    w('<mxGraphModel dx="1422" dy="762" grid="1" gridSize="10" guides="1" tooltips="1" connect="1" arrows="1" fold="1" page="1" pageScale="1" pageWidth="2200" pageHeight="5200" math="0" shadow="0">')
    w('  <root>')
    w('    <mxCell id="0"/>')
    w('    <mxCell id="1" parent="0"/>')

    # === TITLE ===
    box("SkillEngine \u2014 Complete Skill Execution Architecture\nFrom SKILL.md to Tool Execution | Skills First Agent Framework",
        "text;html=1;fontSize=20;fontStyle=1;align=center;verticalAlign=middle;whiteSpace=wrap;",
        300, 20, 1200, 50)

    # === LEGEND ===
    box("", "rounded=1;whiteSpace=wrap;html=1;fillColor=#f5f5f5;strokeColor=#999;", 40, 20, 220, 110)
    box("Legend", "text;html=1;fontSize=12;fontStyle=1;", 120, 23, 50, 20)
    for txt in ["Blue = Process / Data", "Yellow = Decision / Filter", "Purple = EventBus Hook", "Orange = Tool Execution"]:
        box(txt, "text;html=1;fontSize=10;align=left;", 50, 45 + ["Blue = Process / Data", "Yellow = Decision / Filter", "Purple = EventBus Hook", "Orange = Tool Execution"].index(txt)*18, 150, 16)

    # PHASE 1: SKILL DEFINITION
    box("", "rounded=1;whiteSpace=wrap;html=1;fillColor=#F0F4FF;strokeColor=#6c8ebf;strokeWidth=2;dashed=1;dashPattern=8 4;", 30, 150, 1740, 380)
    box("Phase 1: Skill Definition Layer", "text;html=1;fontSize=16;fontStyle=1;align=left;", 50, 155, 300, 25)

    skillmd = box(
        "skills/hello/SKILL.md\n" + SEP + "\n---\nname: hello\ndescription: \"Greeting skill\"\nmetadata:\n  emoji: wave\n  primary_env: \"API_KEY\"\n  requires:\n    bins: [\"node\"]\n    env: [\"TOKEN\"]\n    os: [\"darwin\"]\nuser-invocable: true\n---\n# Skill Content (Markdown)",
        "shape=document;whiteSpace=wrap;html=1;boundedLbl=1;backgroundOutline=1;size=0.05;fillColor=#dae8fc;strokeColor=#6c8ebf;align=left;verticalAlign=top;spacingLeft=8;spacingTop=4;fontSize=11;fontFamily=Courier New;",
        50, 190, 260, 280)

    skill_model = box(
        "Skill (models.py)\n" + SEP + "\nname: str\ndescription: str\ncontent: str  (MD prompt)\nfile_path: Path\nsource: SkillSource\nmetadata: SkillMetadata\nactions: dict[str, SkillAction]\nallowed_tools: list[str]\nmodel: str | None\ncontext: \"fork\" | None\nhooks: dict[str, str]",
        "rounded=0;whiteSpace=wrap;html=1;fillColor=#dae8fc;strokeColor=#6c8ebf;align=left;verticalAlign=top;spacingLeft=8;spacingTop=4;fontSize=11;fontFamily=Courier New;",
        430, 190, 230, 230)

    skill_meta = box(
        "SkillMetadata\n" + SEP + "\nalways: bool\nprimary_env: str\nemoji: str\nrequires: SkillRequirements\ninvocation: InvocationPolicy\ninstall: list[InstallSpec]\ntags: list[str]",
        "rounded=0;whiteSpace=wrap;html=1;fillColor=#d5e8d4;strokeColor=#82b366;align=left;verticalAlign=top;spacingLeft=8;spacingTop=4;fontSize=11;fontFamily=Courier New;",
        780, 190, 240, 170)

    skill_reqs = box(
        "SkillRequirements\n" + "\u2500"*17 + "\nbins: list[str]  (ALL must exist)\nany_bins: list  (ONE+ must exist)\nenv: list[str]  (ALL must be set)\nconfig: list    (files must exist)\nos: list[str]   (platform match)",
        "rounded=0;whiteSpace=wrap;html=1;fillColor=#d5e8d4;strokeColor=#82b366;align=left;verticalAlign=top;spacingLeft=8;spacingTop=4;fontSize=11;fontFamily=Courier New;",
        1140, 190, 270, 140)

    skill_action = box(
        "SkillAction\n" + "\u2500"*10 + "\nname / script / description\nparams: list[ActionParam]\noutput: text|json|file",
        "rounded=0;whiteSpace=wrap;html=1;fillColor=#d5e8d4;strokeColor=#82b366;align=left;verticalAlign=top;spacingLeft=8;spacingTop=4;fontSize=11;fontFamily=Courier New;",
        780, 380, 240, 100)

    box("SkillSource\nBUNDLED | MANAGED\nWORKSPACE | PLUGIN | EXTRA",
        "shape=note;whiteSpace=wrap;html=1;size=14;fillColor=#fff2cc;strokeColor=#d6b656;fontSize=10;align=left;spacingLeft=6;", 1140, 360, 210, 60)

    box("content_hash() = SHA-256\nfor snapshot cache invalidation",
        "shape=note;whiteSpace=wrap;html=1;size=14;fillColor=#fff2cc;strokeColor=#d6b656;fontSize=10;align=left;spacingLeft=6;", 430, 440, 220, 50)

    edge("edgeStyle=orthogonalEdgeStyle;rounded=1;strokeColor=#6c8ebf;", skillmd, skill_model,
         label="MarkdownSkillLoader\nparse YAML frontmatter", label_style="edgeLabel;html=1;fontSize=9;fontStyle=2;fontColor=#6c8ebf;")
    edge("rounded=0;strokeColor=#82b366;exitX=1;exitY=0.3;exitDx=0;exitDy=0;entryX=0;entryY=0.5;entryDx=0;entryDy=0;", skill_model, skill_meta)
    edge("rounded=0;strokeColor=#82b366;exitX=1;exitY=0.5;exitDx=0;exitDy=0;entryX=0;entryY=0.5;entryDx=0;entryDy=0;", skill_meta, skill_reqs)
    edge("rounded=0;strokeColor=#82b366;exitX=1;exitY=0.75;exitDx=0;exitDy=0;entryX=0;entryY=0.5;entryDx=0;entryDy=0;", skill_model, skill_action)

    # PHASE 2: LOADING PIPELINE
    box("", "rounded=1;whiteSpace=wrap;html=1;fillColor=#F0FFF0;strokeColor=#82b366;strokeWidth=2;dashed=1;dashPattern=8 4;", 30, 555, 1740, 200)
    box("Phase 2: Loading Pipeline", "text;html=1;fontSize=16;fontStyle=1;align=left;", 50, 560, 280, 25)

    dirs = box("Skill Directories\n" + SEP + "\n1. BUNDLED  (framework)\n2. MANAGED  (packages)\n3. WORKSPACE (./skills/)\n\nLater overrides earlier\nconfig.merge_dirs()",
        "rounded=1;whiteSpace=wrap;html=1;fillColor=#d5e8d4;strokeColor=#82b366;align=left;spacingLeft=8;fontSize=11;", 50, 595, 240, 140)

    loader = box("MarkdownSkillLoader\n" + "\u2500"*21 + "\nload_directory(path, source)\n--> list[SkillEntry]\n\nScans */SKILL.md pattern\nParses YAML frontmatter\nExtracts Markdown content",
        "rounded=1;whiteSpace=wrap;html=1;fillColor=#d5e8d4;strokeColor=#82b366;align=left;spacingLeft=8;fontSize=11;", 400, 595, 260, 140)

    dedup = box("De-duplication\n" + "\u2500"*13 + "\nBy skill.name\nWORKSPACE > MANAGED > BUNDLED",
        "rounded=1;whiteSpace=wrap;html=1;fillColor=#d5e8d4;strokeColor=#82b366;align=left;spacingLeft=8;fontSize=11;", 770, 610, 260, 80)

    loaded = box("list[Skill]\n(unfiltered)", "rounded=1;whiteSpace=wrap;html=1;fillColor=#dae8fc;strokeColor=#6c8ebf;fontSize=13;fontStyle=1;", 1140, 620, 160, 55)
    box("SkillEntry: skill | load_error\nErrors logged, not fatal",
        "shape=note;whiteSpace=wrap;html=1;size=14;fillColor=#fff2cc;strokeColor=#d6b656;fontSize=10;align=left;spacingLeft=6;", 1400, 620, 220, 45)

    edge("edgeStyle=orthogonalEdgeStyle;rounded=1;strokeColor=#82b366;strokeWidth=2;", dirs, loader)
    edge("edgeStyle=orthogonalEdgeStyle;rounded=1;strokeColor=#82b366;strokeWidth=2;", loader, dedup)
    edge("edgeStyle=orthogonalEdgeStyle;rounded=1;strokeColor=#82b366;strokeWidth=2;", dedup, loaded)

    # PHASE 3: 9-LEVEL FILTER
    box("", "rounded=1;whiteSpace=wrap;html=1;fillColor=#FFFFF0;strokeColor=#d6b656;strokeWidth=2;dashed=1;dashPattern=8 4;", 30, 780, 1740, 520)
    box("Phase 3: DefaultSkillFilter \u2014 9-Level Eligibility Cascade\nShort-circuit on first failure, returns ineligibility reason",
        "text;html=1;fontSize=16;fontStyle=1;align=left;", 50, 785, 600, 40)
    box("FilterContext\nplatform: sys.platform\nenv_vars: os.environ.keys()",
        "rounded=1;whiteSpace=wrap;html=1;fillColor=#e1d5e7;strokeColor=#9673a6;fontSize=10;", 50, 840, 210, 55)

    filters_data = [
        "1. Disabled Check \u2014 config.entries[key].enabled==false?",
        "2. Exclusion List \u2014 skill.name in config.exclude_skills?",
        "3. Always Flag \u2014 metadata.always==true? FORCE INCLUDE",
        "4. Bundled Allowlist \u2014 source==BUNDLED and not in allow_bundled?",
        "5. Binary Requirements \u2014 ALL bins exist?  shutil.which()",
        "6. Any-Bins \u2014 AT LEAST ONE of any_bins exists?",
        "7. Environment Variables \u2014 ALL env vars present?",
        "8. OS Compatibility \u2014 current platform in requires.os?",
        "9. Config File Paths \u2014 ALL config files exist?  Path.exists()",
    ]
    fy = 840
    filter_ids = []
    for i, ftxt in enumerate(filters_data):
        fill = "#d5e8d4" if i == 2 else "#fff2cc"
        stroke = "#82b366" if i == 2 else "#d6b656"
        fid = box(ftxt, f"rounded=1;whiteSpace=wrap;html=1;fillColor={fill};strokeColor={stroke};fontSize=11;align=left;spacingLeft=10;", 350, fy, 420, 30)
        filter_ids.append(fid)
        fy += 40

    for i in range(len(filter_ids) - 1):
        edge("exitX=0.5;exitY=1;exitDx=0;exitDy=0;entryX=0.5;entryY=0;entryDx=0;entryDy=0;strokeColor=#d6b656;", filter_ids[i], filter_ids[i+1])

    fail = box("FILTERED OUT\nShort-circuit!\nreason logged", "rounded=1;whiteSpace=wrap;html=1;fillColor=#f8cecc;strokeColor=#b85450;fontSize=10;", 870, 965, 130, 55)
    edge("exitX=1;exitY=0.5;exitDx=0;exitDy=0;entryX=0;entryY=0.5;entryDx=0;entryDy=0;strokeColor=#b85450;dashed=1;", filter_ids[3], fail,
         label="FAIL", label_style="edgeLabel;html=1;fontSize=9;fontColor=#CC0000;")

    eligible = box("list[Skill] (Eligible)\nAll 9 checks passed", "rounded=1;whiteSpace=wrap;html=1;fillColor=#d5e8d4;strokeColor=#82b366;fontSize=13;fontStyle=1;", 420, 1140, 280, 50)
    edge("exitX=0.5;exitY=1;exitDx=0;exitDy=0;entryX=0.5;entryY=0;entryDx=0;entryDy=0;strokeColor=#82b366;strokeWidth=2;", filter_ids[-1], eligible)

    box("Visibility Split\n" + SEP + "\neligible = all passing skills\nvisible = eligible minus disable_model_invocation\n\nOnly visible skills enter system prompt.\nHidden skills remain callable via dispatch.",
        "shape=note;whiteSpace=wrap;html=1;size=14;fillColor=#fff2cc;strokeColor=#d6b656;fontSize=10;align=left;spacingLeft=6;", 810, 1120, 300, 120)

    # PHASE 4: SNAPSHOT & PROMPT
    box("", "rounded=1;whiteSpace=wrap;html=1;fillColor=#FFF0F5;strokeColor=#d6336c;strokeWidth=2;dashed=1;dashPattern=8 4;", 30, 1270, 1740, 330)
    box("Phase 4: Snapshot Creation and System Prompt Assembly", "text;html=1;fontSize=16;fontStyle=1;align=left;", 50, 1275, 500, 25)

    snapshot = box("SkillSnapshot\n" + SEP + "\nskills: list[Skill]\nprompt: str (pre-formatted)\nversion: int (cache key)\ntimestamp: float\nsource_dirs: list[Path]\n\nImmutable point-in-time view\ncontent_hash for invalidation",
        "rounded=1;whiteSpace=wrap;html=1;fillColor=#e6d0de;strokeColor=#d6336c;align=left;spacingLeft=8;fontSize=11;", 50, 1310, 250, 185)
    fmt = box("format_prompt(visible_skills)", "rounded=1;whiteSpace=wrap;html=1;fillColor=#e6d0de;strokeColor=#d6336c;fontSize=13;fontStyle=1;", 400, 1315, 280, 35)

    fmt_xml = box("XML (Default)\n" + "\u2500"*10 + "\n<skills>\n  <skill>\n    <name>...\n    <description>...\n  </skill>\n</skills>",
        "rounded=1;whiteSpace=wrap;html=1;fillColor=#dae8fc;strokeColor=#6c8ebf;align=left;spacingLeft=6;fontSize=10;fontFamily=Courier New;", 370, 1370, 150, 140)
    fmt_md = box("Markdown\n" + "\u2500"*8 + "\n## Available Skills\n- **name**: desc",
        "rounded=1;whiteSpace=wrap;html=1;fillColor=#d5e8d4;strokeColor=#82b366;align=left;spacingLeft=6;fontSize=10;fontFamily=Courier New;", 530, 1370, 160, 80)
    box('JSON\n' + "\u2500"*5 + '\n[{"name":..}]',
        "rounded=1;whiteSpace=wrap;html=1;fillColor=#fff2cc;strokeColor=#d6b656;align=left;spacingLeft=6;fontSize=10;fontFamily=Courier New;", 530, 1460, 160, 55)

    sys_prompt = box("build_system_prompt()\n" + "\u2500"*20 + "\nLayer 1: Base system prompt (config)\nLayer 2: Context files (AGENTS.md / CLAUDE.md)\nLayer 3: Skills prompt (with token budget)\nLayer 4: User-invocable hints (/skill-name)\n\nBudget truncation applied to Layer 3\nif prompt exceeds limit",
        "rounded=1;whiteSpace=wrap;html=1;fillColor=#e6d0de;strokeColor=#d6336c;align=left;spacingLeft=8;fontSize=11;", 810, 1310, 350, 170)
    box("Prompt Caching\n" + SEP + "\nAnthropic: ephemeral cache_control\nOpenAI: prompt_cache_key\nSnapshot version for invalidation",
        "shape=note;whiteSpace=wrap;html=1;size=14;fillColor=#fff2cc;strokeColor=#d6b656;fontSize=10;align=left;spacingLeft=6;", 810, 1500, 270, 90)

    edge("edgeStyle=orthogonalEdgeStyle;rounded=1;strokeColor=#d6336c;strokeWidth=2;", snapshot, fmt)
    edge("rounded=0;strokeColor=#d6336c;", fmt, fmt_xml)
    edge("rounded=0;strokeColor=#d6336c;", fmt, fmt_md)

    # PHASE 5: REACT AGENT LOOP
    box("", "rounded=1;whiteSpace=wrap;html=1;fillColor=#FFF5F5;strokeColor=#b85450;strokeWidth=2;dashed=1;dashPattern=8 4;", 30, 1630, 1740, 1850)
    box("Phase 5: AgentRunner ReAct Loop \u2014 observe / think / act / learn / repeat (max_turns)",
        "text;html=1;fontSize=16;fontStyle=1;align=left;", 50, 1635, 700, 25)

    user_input = box("User Input", "rounded=1;whiteSpace=wrap;html=1;fillColor=#dae8fc;strokeColor=#6c8ebf;fontSize=14;fontStyle=1;", 530, 1670, 200, 36)
    ev_input = box("INPUT", "shape=hexagon;perimeter=hexagonPerimeter2;whiteSpace=wrap;html=1;fixedSize=1;size=12;fillColor=#e1d5e7;strokeColor=#9673a6;fontSize=11;fontStyle=1;", 540, 1720, 180, 35)
    box("Can transform input or\nshort-circuit (action=handled)", "text;html=1;fontSize=9;align=left;", 740, 1720, 200, 30)

    slash_check = box("/skill-name ?", "rhombus;whiteSpace=wrap;html=1;fillColor=#fff2cc;strokeColor=#d6b656;fontSize=12;fontStyle=1;", 560, 1775, 140, 65)
    slash_proc = box("Skill Invocation\n" + SEP + "\nregex: ^/(\\S+)\n$ARGUMENTS substitution\n$1..$N positional args\n!`cmd` dynamic content\nInject skill content",
        "rounded=1;whiteSpace=wrap;html=1;fillColor=#fff2cc;strokeColor=#d6b656;align=left;spacingLeft=6;fontSize=10;", 790, 1770, 210, 120)
    edge("exitX=1;exitY=0.5;exitDx=0;exitDy=0;entryX=0;entryY=0.3;entryDx=0;entryDy=0;strokeColor=#d6b656;", slash_check, slash_proc,
         label="Yes", label_style="edgeLabel;html=1;fontSize=9;")

    ev_start = box("AGENT_START", "shape=hexagon;perimeter=hexagonPerimeter2;whiteSpace=wrap;html=1;fixedSize=1;size=12;fillColor=#e1d5e7;strokeColor=#9673a6;fontSize=11;fontStyle=1;", 530, 1865, 200, 35)
    box("user_input, system_prompt, model", "text;html=1;fontSize=9;align=left;", 750, 1868, 200, 20)

    box("", "rounded=1;whiteSpace=wrap;html=1;fillColor=none;strokeColor=#b85450;strokeWidth=3;", 60, 1925, 1200, 1370)
    box("LOOP: for turn in range(max_turns)", "text;html=1;fontSize=13;fontStyle=1;fontColor=#b85450;align=left;", 80, 1930, 350, 22)

    abort = box("ABORT check\nabort_signal.is_set()?", "rhombus;whiteSpace=wrap;html=1;fillColor=#f8cecc;strokeColor=#b85450;fontSize=10;", 555, 1965, 150, 60)
    steer = box("STEERING drain (non-blocking)", "rounded=1;whiteSpace=wrap;html=1;fillColor=#f8cecc;strokeColor=#b85450;fontSize=10;", 545, 2040, 170, 28)

    ev_turn_s = box("TURN_START", "shape=hexagon;perimeter=hexagonPerimeter2;whiteSpace=wrap;html=1;fixedSize=1;size=12;fillColor=#e1d5e7;strokeColor=#9673a6;fontSize=10;fontStyle=1;", 540, 2085, 180, 30)
    ev_ctx = box("CONTEXT_TRANSFORM", "shape=hexagon;perimeter=hexagonPerimeter2;whiteSpace=wrap;html=1;fixedSize=1;size=12;fillColor=#e1d5e7;strokeColor=#9673a6;fontSize=10;fontStyle=1;", 520, 2132, 220, 30)
    box("Handlers can prune/inject messages\nMemory injection, context filtering", "text;html=1;fontSize=9;align=left;", 760, 2128, 220, 30)

    compact = box("Context Compaction\nif should_compact()", "rounded=1;whiteSpace=wrap;html=1;fillColor=#f8cecc;strokeColor=#b85450;fontSize=10;", 550, 2180, 160, 35)
    box("Compaction Strategies\n" + SEP2 + "\nTokenBudgetCompactor:\n  Remove oldest to fit budget\n  Preserve tool call/result pairs\nSlidingWindowCompactor:\n  Keep last N turns\nToken estimate: len(text) // 4",
        "shape=note;whiteSpace=wrap;html=1;size=14;fillColor=#fff2cc;strokeColor=#d6b656;fontSize=10;align=left;spacingLeft=6;", 80, 2150, 250, 140)

    llm_call = box("_call_llm(messages)\nLLM API call via Adapter",
        "rounded=1;whiteSpace=wrap;html=1;fillColor=#dae8fc;strokeColor=#6c8ebf;fontSize=13;fontStyle=1;strokeWidth=2;", 510, 2235, 240, 45)
    box("Cross-Provider Adapters\n" + SEP3 + "\nOpenAIAdapter / AnthropicAdapter\ntransform_messages():\n  Normalize tool_call IDs (SHA-256)\n  Convert thinking blocks\n  Synthetic empty tool results\n  ThinkingLevel mapping",
        "shape=note;whiteSpace=wrap;html=1;size=14;fillColor=#fff2cc;strokeColor=#d6b656;fontSize=10;align=left;spacingLeft=6;", 80, 2310, 270, 140)

    ev_turn_e = box("TURN_END", "shape=hexagon;perimeter=hexagonPerimeter2;whiteSpace=wrap;html=1;fixedSize=1;size=12;fillColor=#e1d5e7;strokeColor=#9673a6;fontSize=10;fontStyle=1;", 540, 2300, 180, 30)
    box("has_tool_calls, content, tool_call_count", "text;html=1;fontSize=9;align=left;", 740, 2300, 250, 20)

    has_tc = box("has tool_calls?", "rhombus;whiteSpace=wrap;html=1;fillColor=#fff2cc;strokeColor=#d6b656;fontSize=12;fontStyle=1;", 560, 2350, 140, 60)
    ret_resp = box("Return Response\ntext_content to user", "rounded=1;whiteSpace=wrap;html=1;fillColor=#d5e8d4;strokeColor=#82b366;fontSize=11;", 340, 2358, 170, 40)
    edge("exitX=0;exitY=0.5;exitDx=0;exitDy=0;entryX=1;entryY=0.5;entryDx=0;entryDy=0;strokeColor=#82b366;", has_tc, ret_resp,
         label="No", label_style="edgeLabel;html=1;fontSize=9;")

    tc_border = box("", "rounded=1;whiteSpace=wrap;html=1;fillColor=none;strokeColor=#d6b656;strokeWidth=2;dashed=1;", 400, 2440, 820, 810)
    box("FOR each tool_call in response.tool_calls", "text;html=1;fontSize=12;fontStyle=1;fontColor=#d6b656;align=left;", 420, 2445, 400, 20)
    edge("exitX=0.5;exitY=1;exitDx=0;exitDy=0;entryX=0.5;entryY=0;entryDx=0;entryDy=0;strokeColor=#d6b656;strokeWidth=2;", has_tc, tc_border,
         label="Yes", label_style="edgeLabel;html=1;fontSize=9;")

    ev_before = box("BEFORE_TOOL_CALL", "shape=hexagon;perimeter=hexagonPerimeter2;whiteSpace=wrap;html=1;fixedSize=1;size=12;fillColor=#e1d5e7;strokeColor=#9673a6;fontSize=11;fontStyle=1;", 530, 2480, 220, 35)
    box("ToolCallEventResult\n" + SEP2 + "\nblock: bool  (prevent execution)\nreason: str  (sent back to LLM)\nmodified_args: dict\n\nGuard rails at priority=0\nUser handlers at higher priority",
        "shape=note;whiteSpace=wrap;html=1;size=14;fillColor=#fff2cc;strokeColor=#d6b656;fontSize=10;align=left;spacingLeft=6;", 850, 2470, 260, 140)

    blocked = box("blocked?", "rhombus;whiteSpace=wrap;html=1;fillColor=#fff2cc;strokeColor=#d6b656;fontSize=11;fontStyle=1;", 590, 2530, 100, 55)
    blocked_yes = box("[Blocked] reason\nas tool result", "rounded=1;whiteSpace=wrap;html=1;fillColor=#f8cecc;strokeColor=#b85450;fontSize=9;", 430, 2540, 120, 35)
    edge("exitX=0;exitY=0.5;exitDx=0;exitDy=0;entryX=1;entryY=0.5;entryDx=0;entryDy=0;strokeColor=#b85450;", blocked, blocked_yes,
         label="Yes", label_style="edgeLabel;html=1;fontSize=9;")

    exec_tool = box("_execute_tool(tool_call)\nRoute by tool name | on_output callback",
        "rounded=1;whiteSpace=wrap;html=1;fillColor=#ffe6cc;strokeColor=#d79b00;fontSize=12;fontStyle=1;strokeWidth=2;", 495, 2610, 290, 40)
    edge("exitX=0.5;exitY=1;exitDx=0;exitDy=0;entryX=0.5;entryY=0;entryDx=0;entryDy=0;strokeColor=#d79b00;strokeWidth=2;", blocked, exec_tool,
         label="No", label_style="edgeLabel;html=1;fontSize=9;")

    box("7 Tool Dispatch Paths:", "text;html=1;fontSize=11;fontStyle=1;fontColor=#d79b00;align=left;", 420, 2665, 200, 18)

    tools_data = [
        ("execute\n" + "\u2500"*9 + "\nSingle shell cmd\nBashRuntime\nasyncio subprocess", 420, 2690, 110, 80),
        ("execute_script\n" + "\u2500"*10 + "\nMulti-line script\nBashRuntime", 540, 2690, 110, 80),
        ("write\n" + "\u2500"*6 + "\nPath.write_text()\nmkdir parents", 660, 2690, 100, 80),
        ("read\n" + "\u2500"*6 + "\nPath.read_text()\n100KB truncation", 770, 2690, 110, 80),
        ("skill\n" + "\u2500"*6 + "\nLoad skill content\n$ARGUMENTS subst\n!`cmd` preprocess\nmodel override\ncontext=fork subagent", 420, 2785, 140, 110),
        ("skill:action\n" + "\u2500"*10 + "\nParse name:action\n_build_action_args\nengine.execute_action()\nCLI subprocess", 575, 2785, 140, 100),
        ("extension tool\n" + "\u2500"*10 + "\nExtension.handle_tool_call()\nMemory / MCP integration", 730, 2785, 170, 80),
    ]
    for tval, tx, ty, tw, th in tools_data:
        box(tval, "rounded=1;whiteSpace=wrap;html=1;fillColor=#ffe6cc;strokeColor=#d79b00;fontSize=9;align=left;spacingLeft=4;", tx, ty, tw, th)

    box("with engine.env_context():  Thread-safe (ContextVar + Lock)\nBackup -> Apply overrides -> Execute -> Restore  |  primary_env: API_KEY injection",
        "rounded=1;whiteSpace=wrap;html=1;fillColor=#f5f5f5;strokeColor=#666;dashed=1;dashPattern=5 5;align=left;spacingLeft=6;fontSize=10;", 420, 2905, 480, 45)
    box("Forked SubAgent\n" + SEP2 + "\ncontext=\"fork\" triggers isolation\nchild = AgentRunner(child_config)\nSeparate system_prompt = skill.content\nFiltered allowed_tools\nIndependent conversation context\nReturns response.text_content",
        "shape=note;whiteSpace=wrap;html=1;size=14;fillColor=#fff2cc;strokeColor=#d6b656;fontSize=10;align=left;spacingLeft=6;", 80, 2730, 270, 140)

    ev_after = box("AFTER_TOOL_RESULT", "shape=hexagon;perimeter=hexagonPerimeter2;whiteSpace=wrap;html=1;fixedSize=1;size=12;fillColor=#e1d5e7;strokeColor=#9673a6;fontSize=11;fontStyle=1;", 525, 2970, 230, 35)
    box("modified_result: str \u2014 can replace\noutput before it enters conversation", "text;html=1;fontSize=9;align=left;", 780, 2970, 230, 30)

    append = box('Append Tool Result: AgentMessage(role="tool", content, tool_call_id)',
        "rounded=1;whiteSpace=wrap;html=1;fillColor=#f8cecc;strokeColor=#b85450;fontSize=10;", 510, 3025, 260, 35)
    more_tc = box("more\ntool_calls?", "rhombus;whiteSpace=wrap;html=1;fillColor=#fff2cc;strokeColor=#d6b656;fontSize=10;fontStyle=1;", 585, 3080, 110, 60)

    edge("edgeStyle=orthogonalEdgeStyle;rounded=1;strokeColor=#d6b656;exitX=1;exitY=0.5;exitDx=0;exitDy=0;entryX=1;entryY=0.5;entryDx=0;entryDy=0;curved=1;",
         more_tc, ev_before, pts=[(1100, 3110), (1100, 2498)],
         label="Yes", label_style="edgeLabel;html=1;fontSize=9;")
    edge("edgeStyle=orthogonalEdgeStyle;rounded=1;strokeColor=#b85450;strokeWidth=2;exitX=0.5;exitY=1;exitDx=0;exitDy=0;curved=1;",
         more_tc, abort, pts=[(640, 3190), (170, 3190), (170, 1995), (555, 1995)],
         label="No, next turn", label_style="edgeLabel;html=1;fontSize=9;")

    # Flow arrows
    flow = [
        (user_input, ev_input), (ev_input, slash_check),
        (slash_check, ev_start), (ev_start, abort),
        (abort, steer), (steer, ev_turn_s),
        (ev_turn_s, ev_ctx), (ev_ctx, compact),
        (compact, llm_call), (llm_call, ev_turn_e),
        (ev_turn_e, has_tc), (ev_before, blocked),
        (exec_tool, ev_after), (ev_after, append),
        (append, more_tc),
    ]
    for src, tgt in flow:
        edge("exitX=0.5;exitY=1;exitDx=0;exitDy=0;entryX=0.5;entryY=0;entryDx=0;entryDy=0;strokeColor=#666;", src, tgt)

    box("Streaming Architecture\n" + SEP2 + "\nchat()  -> sync AgentMessage\nchat_stream()  -> text_delta only\nchat_stream_events()  -> fine-grained:\n  text/thinking start/delta/end\n  tool_call start/delta/end\n  tool_result, tool_output\n  turn_start/end, done, error",
        "shape=note;whiteSpace=wrap;html=1;size=14;fillColor=#fff2cc;strokeColor=#d6b656;fontSize=10;align=left;spacingLeft=6;", 80, 2910, 270, 160)

    ev_end = box("AGENT_END", "shape=hexagon;perimeter=hexagonPerimeter2;whiteSpace=wrap;html=1;fixedSize=1;size=12;fillColor=#e1d5e7;strokeColor=#9673a6;fontSize=12;fontStyle=1;", 530, 3240, 200, 35)
    box("finally block (always runs)\n" + SEP2 + "\ntotal_turns, finish_reason, error\nfinish_reason:\n  \"complete\" | \"max_turns\"\n  | \"aborted\" | \"error\"",
        "shape=note;whiteSpace=wrap;html=1;size=14;fillColor=#fff2cc;strokeColor=#d6b656;fontSize=10;align=left;spacingLeft=6;", 830, 3230, 240, 100)

    final_resp = box('Final Response: AgentMessage(role="assistant", content, finish_reason)',
        "rounded=1;whiteSpace=wrap;html=1;fillColor=#d5e8d4;strokeColor=#82b366;fontSize=13;fontStyle=1;strokeWidth=2;", 460, 3310, 340, 45)
    edge("exitX=0.5;exitY=1;exitDx=0;exitDy=0;entryX=0.5;entryY=0;entryDx=0;entryDy=0;strokeColor=#82b366;strokeWidth=2;", ev_end, final_resp)

    # PHASE 6: INFRASTRUCTURE
    box("", "rounded=1;whiteSpace=wrap;html=1;fillColor=#FFF8F0;strokeColor=#d79b00;strokeWidth=2;dashed=1;dashPattern=8 4;", 30, 3510, 1740, 280)
    box("Phase 6: Runtime Infrastructure and Supporting Systems", "text;html=1;fontSize=16;fontStyle=1;align=left;", 50, 3515, 500, 25)

    box("BashRuntime\n" + SEP + "\nexecute(cmd, cwd, env, timeout)\nexecute_script(script, ...)\nasyncio.create_subprocess_shell\nStreaming on_output callback\nAbort signal + Timeout\nMax output: 1MB (configurable)",
        "rounded=1;whiteSpace=wrap;html=1;fillColor=#ffe6cc;strokeColor=#d79b00;align=left;spacingLeft=8;fontSize=11;", 50, 3555, 280, 150)
    box("env_context() Manager\n" + SEP3 + "\n1. Lock (threading.Lock)\n2. Backup (ContextVar per-thread)\n3. _apply_env_overrides():\n   skill_config.env -> os.environ\n   api_key -> primary_env var\n4. Execute within context\n5. Restore backup on exit",
        "rounded=1;whiteSpace=wrap;html=1;fillColor=#f5f5f5;strokeColor=#666;align=left;spacingLeft=8;fontSize=11;", 380, 3555, 280, 170)
    box("Extension System\n" + SEP2 + "\nLifecycle Hooks:\n  on_agent_start/end\n  on_context_transform\nTool Registration:\n  get_tools() -> definitions\n  handle_tool_call() -> execute\nExample: OpenViking Memory\n  (4 tools + 3 lifecycle hooks)",
        "rounded=1;whiteSpace=wrap;html=1;fillColor=#e1d5e7;strokeColor=#9673a6;align=left;spacingLeft=8;fontSize=11;", 710, 3555, 280, 180)
    box("Session and Adapters\n" + "\u2500"*17 + "\nSession: JSONL append-only\n  Tree branching (fork/navigate)\n  Conversation replay\nAdapters: OpenAI / Anthropic\n  tool_call ID normalization\n  (SHA-256 if >64 chars)\n  ThinkingLevel mapping",
        "rounded=1;whiteSpace=wrap;html=1;fillColor=#dae8fc;strokeColor=#6c8ebf;align=left;spacingLeft=8;fontSize=11;", 1040, 3555, 280, 170)

    # EVENTBUS CHAIN
    box("", "rounded=1;whiteSpace=wrap;html=1;fillColor=#F5F0FF;strokeColor=#9673a6;strokeWidth=2;dashed=1;dashPattern=8 4;", 30, 3820, 1740, 140)
    box("EventBus Lifecycle \u2014 Complete Hook Chain", "text;html=1;fontSize=16;fontStyle=1;align=left;", 50, 3825, 400, 25)

    events_data = [
        ("INPUT", 50, 90), ("AGENT_START", 190, 120),
        ("TURN_START", 360, 110), ("CONTEXT_TRANSFORM", 520, 150),
        ("TURN_END", 720, 100), ("BEFORE_TOOL_CALL", 870, 140),
        ("AFTER_TOOL_RESULT", 1060, 140), ("AGENT_END", 1250, 110),
    ]
    event_ids = []
    for eval_, ex, ew in events_data:
        eid = box(eval_, "shape=hexagon;perimeter=hexagonPerimeter2;whiteSpace=wrap;html=1;fixedSize=1;size=8;fillColor=#e1d5e7;strokeColor=#9673a6;fontSize=9;fontStyle=1;", ex, 3865, ew, 28)
        event_ids.append(eid)

    for i in range(len(event_ids) - 1):
        edge("strokeColor=#9673a6;strokeWidth=2;", event_ids[i], event_ids[i+1])

    box("Priority: lower runs first (guards=0, user=higher) | Register: @bus.on(name, priority) | BEFORE_TOOL_CALL can block/modify | AFTER_TOOL_RESULT can replace | CONTEXT_TRANSFORM can inject",
        "text;html=1;fontSize=10;align=left;whiteSpace=wrap;", 50, 3910, 1200, 30)

    # PIPELINE SUMMARY
    box("", "rounded=1;whiteSpace=wrap;html=1;fillColor=#E8F5E9;strokeColor=#2E7D32;strokeWidth=2;", 30, 3990, 1740, 55)
    box("SKILL.md (YAML+MD) -> MarkdownSkillLoader -> DefaultSkillFilter (9-level) -> SkillSnapshot -> format_prompt (XML/MD/JSON) -> System Prompt (4 layers) -> ReAct Loop -> _execute_tool (7 paths) -> BashRuntime + env_context -> EventBus hooks",
        "text;html=1;fontSize=12;fontStyle=1;align=center;verticalAlign=middle;whiteSpace=wrap;", 40, 3995, 1720, 45)

    w('  </root>')
    w('</mxGraphModel>')

    # Build raw XML
    raw_xml = "\n".join(lines)

    # Compress: deflate -> base64 -> urlencode
    encoded = compress_xml(raw_xml)

    # Write the .drawio file with compressed content
    output = f'''<mxfile host="app.diagrams.net" modified="2026-02-20T00:00:00.000Z" agent="5.0" etag="gen" version="24.2.0" type="device">
  <diagram name="Skill Execution Flow" id="exec_flow">{encoded}</diagram>
</mxfile>
'''
    with open("/Users/sawzhang/code/agent-skills-engine/docs/skillengine-execution-flow.drawio", "w") as f:
        f.write(output)
    print("Done! skillengine-execution-flow.drawio regenerated (compressed format)")

if __name__ == "__main__":
    main()
