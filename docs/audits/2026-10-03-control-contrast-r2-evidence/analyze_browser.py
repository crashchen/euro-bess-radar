"""Audit saved actual-DOM colors without averaging gradient stops."""
import json
import re
from pathlib import Path

HERE = Path(__file__).parent

def rgb(value):
    numbers = [float(v) for v in re.findall(r'[\d.]+', value)]
    return numbers[:3], numbers[3] if len(numbers) > 3 else 1.0

def lum(values):
    linear = [v / 255 / 12.92 if v / 255 <= .04045 else ((v / 255 + .055) / 1.055) ** 2.4 for v in values]
    return sum(w*v for w,v in zip((.2126,.7152,.0722), linear, strict=True))

def contrast_bound(button):
    stops = [rgb(s) for s in re.findall(r'rgba?\([^)]+\)', button['gradient'])]
    assert len(stops) == 2 and all(alpha == 1 for _,alpha in stops), button
    assert button['filter'] == 'none', button
    # Bounding each channel above both stops bounds sRGB gradient luminance
    # above every interpolated position. No averaged color or antialiased pixel.
    ceiling = [max(color[i] for color,_ in stops) for i in range(3)]
    leaves = button['text'] or [button]
    ratios=[]
    for leaf in leaves:
        foreground,alpha = rgb(leaf['fill'] or leaf['color'])
        assert alpha == 1 and foreground == [255,255,255], leaf
        ratios.append((lum(foreground)+.05)/(lum(ceiling)+.05))
    return min(ratios)

rows=[]
for theme in ['light','dark']:
    for width in [390,960,1280,1440]:
        base=json.loads((HERE/'browser'/f'baseline-{theme}-{width}.json').read_text())
        candidate=json.loads((HERE/'browser'/f'candidate-{theme}-{width}.json').read_text())
        assert candidate['normal']['viewport']['width']==width
        assert candidate['normal']['sidebarExpanded']==('false' if width<600 else 'true')
        assert not candidate['normal']['exceptions'] and not candidate['dataTrust']['exceptions']
        normal=[b for b in candidate['normal']['buttons'] if b['scope']!='sidebar' and not b['disabled']]
        hover=[b for b in candidate['hover'] if not b['disabled']]
        assert len(normal)==6 and len(hover)==6
        assert all(b['hover'] for b in candidate['hover'])
        normal_bound=min(contrast_bound(b) for b in normal)
        hover_bound=min(contrast_bound(b) for b in hover)
        assert min(normal_bound,hover_bound)>=4.5
        disabled=[b for b in candidate['hover'] if b['disabled']]
        assert len(disabled)==2
        for b in disabled:
            assert b['shadow']=='none' and b['filter']=='none'
            assert '35, 42, 54' in b['gradient']
            assert b['gradient'] != normal[0]['gradient']
        assert candidate['dataTrust']['metrics']==base['dataTrust']['metrics']
        assert len(candidate['dataTrust']['metrics'])==4
        assert not any(m['truncated'] for m in candidate['dataTrust']['metrics'])
        fields=['label','uploader','scope','disabled','color','fill','background','gradient','font','weight','opacity']
        def basis(buttons): return [{k:b.get(k) for k in fields} for b in buttons]
        assert basis(candidate['sidebar'])==basis(base['sidebar']), (theme,width,'sidebar changed')
        def uploader_basis(probe):
            return [(u['label'],u['background']['gradient'],[(t['text'],t['color'],t['font']) for t in u['text']]) for u in probe['uploaders']]
        assert uploader_basis(candidate['normal'])==uploader_basis(base['normal']), (theme,width,'uploader instructions changed')
        rows.append({'theme':theme,'width':width,'normal_contrast_lower_bound':normal_bound,
                     'hover_contrast_lower_bound':hover_bound,'enabled_buttons_each_state':6,
                     'actual_hover_checks':8,'disabled_browse_separate':True,'sidebar_unchanged':True,
                     'uploader_instructions_unchanged':True,'metrics_unchanged_and_unclipped':True})
result={'method':'Conservative channel-wise luminance ceiling across two opaque sRGB stops, using actual rendered leaf foreground. Normal and hover captured after real pointer movement; disabled controls excluded from the contrast threshold. Browser/fixture scope, not whole-app WCAG certification.', 'cases':rows}
assert json.loads((HERE/'browser-summary.json').read_text()) == result, 'Saved summary differs from recomputed matrix'
# Reverification leaves the archived summary and all raw evidence unchanged.
print(json.dumps(result,indent=2))
