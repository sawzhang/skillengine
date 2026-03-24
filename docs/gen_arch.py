"""Regenerate architecture.drawio using COMPRESSED format (deflate+base64+urlencode).
This matches the format used by draw.io's native save, proven to work in test-minimal.drawio.
"""
import base64, zlib, urllib.parse
from xml.sax.saxutils import escape

lines = []
def w(s):
    lines.append(s)

def v(text):
    result = escape(text, {'"': "&quot;"})
    result = result.replace("\n", "&#xa;")
    return result

def box(id, val, style, x, y, width, height):
    w(f'    <mxCell id="{id}" value="{v(val)}" style="{style}" vertex="1" parent="1">')
    w(f'      <mxGeometry x="{x}" y="{y}" width="{width}" height="{height}" as="geometry"/>')
    w(f'    </mxCell>')

def edge(id, style, src, tgt, pts=None, label=None, label_style=None):
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
        ls = label_style or "edgeLabel;html=1;fontSize=10;fontStyle=2;"
        w(f'    <mxCell id="{id}_label" value="{v(label)}" style="{ls}" vertex="1" connectable="0" parent="{id}">')
        w(f'      <mxGeometry relative="1" as="geometry"><mxPoint as="offset"/></mxGeometry>')
        w(f'    </mxCell>')

def compress_xml(xml_str):
    """Compress XML: deflate -> base64. Raw base64 output (no URI encoding)."""
    xml_bytes = xml_str.encode("utf-8")
    compressor = zlib.compressobj(zlib.Z_DEFAULT_COMPRESSION, zlib.DEFLATED, -15)
    compressed = compressor.compress(xml_bytes) + compressor.flush()
    return base64.b64encode(compressed).decode("ascii")

def main():
    # Build the mxGraphModel XML (inner content only)
    w('<mxGraphModel dx="1422" dy="762" grid="1" gridSize="10" guides="1" tooltips="1" connect="1" arrows="1" fold="1" page="1" pageScale="1" pageWidth="1600" pageHeight="900" math="0" shadow="0">')
    w('  <root>')
    w('    <mxCell id="0"/>')
    w('    <mxCell id="1" parent="0"/>')

    # Title
    box("2", "SkillEngine \u2014 Agent Skills Engine Architecture",
        "text;html=1;fontSize=20;fontStyle=1;align=center;verticalAlign=middle;", 400, 20, 800, 40)

    # USER / LLM Layer
    box("3", "\U0001f464 User",
        "shape=actor;whiteSpace=wrap;html=1;fillColor=#d5e8d4;strokeColor=#82b366;fontSize=13;fontStyle=1;", 120, 90, 80, 80)
    box("4", "\U0001f916 LLM Provider\n(OpenAI / Anthropic / MiniMax)",
        "rounded=1;whiteSpace=wrap;html=1;fillColor=#dae8fc;strokeColor=#6c8ebf;fontSize=12;fontStyle=1;", 1200, 90, 260, 60)

    # AgentRunner group
    box("5", "",
        "rounded=1;whiteSpace=wrap;html=1;fillColor=#fff2cc;strokeColor=#d6b656;strokeWidth=2;verticalAlign=top;fontSize=14;fontStyle=1;dashed=0;", 280, 80, 860, 280)
    box("6", "AgentRunner  (agent.py)",
        "text;html=1;fontSize=16;fontStyle=1;align=center;", 560, 85, 300, 30)

    # System Prompt Builder
    box("7",
        "build_system_prompt()\n\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\n\u2022 Base prompt\n\u2022 Context files (AGENTS.md)\n\u2022 Skill metadata (name+desc)\n\u2022 Description budget (16K)",
        "rounded=1;whiteSpace=wrap;html=1;fillColor=#f8cecc;strokeColor=#b85450;fontSize=11;align=left;spacingLeft=8;", 300, 120, 220, 110)

    # Skill Tool
    box("8",
        "skill Tool (on-demand)\n\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\n\u2022 Load full SKILL.md content\n\u2022 $ARGUMENTS substitution\n\u2022 !`cmd` dynamic injection\n\u2022 Per-skill model switch\n\u2022 allowed-tools hint\n\u2022 context: fork \u2192 child agent",
        "rounded=1;whiteSpace=wrap;html=1;fillColor=#e1d5e7;strokeColor=#9673a6;fontSize=11;align=left;spacingLeft=8;fontStyle=0;", 540, 120, 230, 130)

    # Other Tools
    box("9",
        "Built-in Tools\n\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\n\u2022 execute (shell cmd)\n\u2022 execute_script\n\u2022 skill:action (deterministic)\n\u2022 Extension tools",
        "rounded=1;whiteSpace=wrap;html=1;fillColor=#d5e8d4;strokeColor=#82b366;fontSize=11;align=left;spacingLeft=8;", 790, 120, 190, 110)

    # Slash Commands
    box("10",
        "/skill-name args\n\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\n\u2022 $ARGUMENTS substitution\n\u2022 context: fork bypass\n\u2022 Dynamic injection",
        "rounded=1;whiteSpace=wrap;html=1;fillColor=#fff2cc;strokeColor=#d6b656;fontSize=11;align=left;spacingLeft=8;", 300, 245, 200, 100)

    # Validation
    box("11",
        "validate_skill()\nname \u226464, [a-z0-9-]\ndesc \u22641024, non-empty",
        "rounded=1;whiteSpace=wrap;html=1;fillColor=#f5f5f5;strokeColor=#666666;fontSize=10;fontColor=#333333;", 540, 270, 180, 60)

    # Fork
    box("12",
        "\U0001f500 context: fork\nChild AgentRunner\n(isolated context)",
        "rounded=1;whiteSpace=wrap;html=1;fillColor=#e1d5e7;strokeColor=#9673a6;fontSize=11;fontStyle=1;", 790, 255, 170, 70)

    # SkillsEngine group
    box("13", "",
        "rounded=1;whiteSpace=wrap;html=1;fillColor=#dae8fc;strokeColor=#6c8ebf;strokeWidth=2;verticalAlign=top;", 280, 400, 860, 200)
    box("14", "SkillsEngine  (engine.py)",
        "text;html=1;fontSize=16;fontStyle=1;align=center;", 560, 405, 300, 30)

    # Loader
    box("15",
        "\U0001f4c4 Loader\nMarkdownSkillLoader\n\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\nParse YAML frontmatter\n+ new fields:\nmodel, context,\nallowed-tools,\nargument-hint, hooks",
        "rounded=1;whiteSpace=wrap;html=1;fillColor=#fff2cc;strokeColor=#d6b656;fontSize=10;align=left;spacingLeft=6;", 300, 440, 170, 140)

    # Filter
    box("16",
        "\U0001f50d Filter\nDefaultSkillFilter\n\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\nbins, env, OS,\nconfig checks\nShort-circuit",
        "rounded=1;whiteSpace=wrap;html=1;fillColor=#fff2cc;strokeColor=#d6b656;fontSize=10;align=left;spacingLeft=6;", 490, 440, 160, 120)

    # Snapshot
    box("17",
        "\U0001f4f8 SkillSnapshot\n\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\nskills: List[Skill]\nprompt: str (metadata only)\nversion + timestamp",
        "rounded=1;whiteSpace=wrap;html=1;fillColor=#f8cecc;strokeColor=#b85450;fontSize=10;align=left;spacingLeft=6;fontStyle=0;", 670, 440, 190, 100)

    # Runtime
    box("18",
        "\u26a1 Runtime\nBashRuntime\n\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\nExecute commands\nTimeout + env injection\nStreaming output",
        "rounded=1;whiteSpace=wrap;html=1;fillColor=#fff2cc;strokeColor=#d6b656;fontSize=10;align=left;spacingLeft=6;", 880, 440, 160, 120)

    # Skill Files
    box("19",
        "\U0001f4c1 skills/\n\u251c\u2500\u2500 pdf/SKILL.md\n\u251c\u2500\u2500 csv/SKILL.md\n\u2514\u2500\u2500 search/SKILL.md",
        "rounded=0;whiteSpace=wrap;html=1;fillColor=#f5f5f5;strokeColor=#666666;fontSize=11;align=left;spacingLeft=10;fontFamily=Courier New;", 60, 460, 180, 80)

    # Skill Model group
    box("20", "",
        "rounded=1;whiteSpace=wrap;html=1;fillColor=#e6d0de;strokeColor=#996185;strokeWidth=2;", 280, 640, 860, 220)
    box("21", "Skill Data Model  (models.py)",
        "text;html=1;fontSize=16;fontStyle=1;align=center;", 560, 645, 300, 30)

    box("22",
        "Skill\n\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\nname: str\ndescription: str\ncontent: str\nfile_path: Path\nbase_dir: Path\nsource: SkillSource\nmetadata: SkillMetadata\nactions: dict[str, SkillAction]",
        "rounded=1;whiteSpace=wrap;html=1;fillColor=#ffffff;strokeColor=#996185;fontSize=10;align=left;spacingLeft=6;fontFamily=Courier New;", 300, 680, 210, 165)

    box("23",
        "Claude Agent Skills Extensions\n\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\nallowed_tools: list[str]\nmodel: str | None\ncontext: str | None\nargument_hint: str | None\nhooks: dict[str, str]",
        "rounded=1;whiteSpace=wrap;html=1;fillColor=#e1d5e7;strokeColor=#9673a6;fontSize=10;align=left;spacingLeft=6;fontFamily=Courier New;fontStyle=1;", 530, 680, 250, 130)

    box("24",
        "SkillMetadata\n\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\nrequires: SkillRequirements\ninvocation: SkillInvocationPolicy\nemoji, tags, primary_env\ninstall: list[SkillInstallSpec]",
        "rounded=1;whiteSpace=wrap;html=1;fillColor=#ffffff;strokeColor=#996185;fontSize=10;align=left;spacingLeft=6;fontFamily=Courier New;", 800, 680, 240, 110)

    # Arrows
    edge("30", "edgeStyle=orthogonalEdgeStyle;rounded=1;", "3", "10",
         label="/pdf report.pdf")
    edge("31", "edgeStyle=orthogonalEdgeStyle;rounded=1;", "9", "4",
         label="API calls\n(tools + system prompt)")
    edge("32", "edgeStyle=orthogonalEdgeStyle;rounded=1;dashed=1;strokeColor=#9673a6;", "4", "8",
         pts=[(1180, 180)],
         label='skill(name="pdf",\narguments="report.pdf")', label_style="edgeLabel;html=1;fontSize=10;fontStyle=2;fontColor=#9673a6;")
    edge("33", "edgeStyle=orthogonalEdgeStyle;rounded=1;dashed=1;strokeColor=#9673a6;", "8", "12")
    edge("34", "edgeStyle=orthogonalEdgeStyle;rounded=1;", "19", "15",
         label="SKILL.md", label_style="edgeLabel;html=1;fontSize=10;")
    edge("35", "edgeStyle=orthogonalEdgeStyle;rounded=1;", "15", "16")
    edge("36", "edgeStyle=orthogonalEdgeStyle;rounded=1;", "16", "17")
    edge("37", "edgeStyle=orthogonalEdgeStyle;rounded=1;strokeColor=#b85450;", "17", "7",
         pts=[(765, 390), (410, 390)],
         label="metadata only\n(name + description)", label_style="edgeLabel;html=1;fontSize=10;fontColor=#b85450;fontStyle=2;")
    edge("38", "edgeStyle=orthogonalEdgeStyle;rounded=1;strokeColor=#9673a6;dashed=1;", "17", "8",
         pts=[(765, 380), (655, 380)],
         label="full content\n(on demand)", label_style="edgeLabel;html=1;fontSize=10;fontColor=#9673a6;fontStyle=2;")

    # Legend
    box("40", "", "rounded=1;whiteSpace=wrap;html=1;fillColor=#f5f5f5;strokeColor=#999999;strokeWidth=1;", 1200, 640, 260, 160)
    box("41", "Legend", "text;html=1;fontSize=13;fontStyle=1;", 1300, 645, 60, 25)
    box("42", "\u2500\u2500 Startup / always", "text;html=1;fontSize=10;align=left;", 1220, 675, 220, 20)
    box("43", "\u254c\u254c On-demand (LLM calls skill tool)", "text;html=1;fontSize=10;align=left;fontColor=#9673a6;", 1220, 700, 220, 20)
    box("44", "\U0001f7e1 Core pipeline", "text;html=1;fontSize=10;align=left;", 1220, 725, 220, 20)
    box("45", "\U0001f7e3 Claude Agent Skills extensions", "text;html=1;fontSize=10;align=left;", 1220, 750, 220, 20)
    box("46", "\U0001f534 Prompt / metadata path", "text;html=1;fontSize=10;align=left;", 1220, 775, 220, 20)

    w('  </root>')
    w('</mxGraphModel>')

    # Build raw XML
    raw_xml = "\n".join(lines)

    # Compress: deflate -> base64 -> urlencode
    encoded = compress_xml(raw_xml)

    # Write the .drawio file with compressed content
    output = f'''<mxfile host="app.diagrams.net" modified="2026-02-20T00:00:00.000Z" agent="5.0" etag="gen" version="24.2.0" type="device">
  <diagram name="SkillEngine Architecture" id="skillengine_arch">{encoded}</diagram>
</mxfile>
'''
    with open("/Users/sawzhang/code/agent-skills-engine/docs/architecture.drawio", "w") as f:
        f.write(output)
    print("Done! architecture.drawio regenerated (compressed format)")

if __name__ == "__main__":
    main()
