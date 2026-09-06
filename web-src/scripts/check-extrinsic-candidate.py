"""Exercise the production standalone handlers in Chromium; API/frame are fixtures."""
import json
from pathlib import Path
from playwright.sync_api import sync_playwright

root = Path(__file__).resolve().parents[1]
source = (root / 'src/extrinsic-legacy.ts').read_text()
ids = ['camera-placeholder', 'mode-chip', 'input-chip', 'error-banner', 'success-banner',
       'frame-meta', 'coordinate-hint', 'result-box', 'image-topic', 'intrinsic-file',
       'pose-prefix', 'output-file']
html = '<canvas id="camera-canvas" width="640" height="480"></canvas>'
html += ''.join('<div id="%s"></div>' % item for item in ids)
html += '<select id="marker-select"></select><table><tbody id="points-body"></tbody></table>'
html += ''.join('<button id="%s-button">%s</button>' % (item, item) for item in ['freeze','live','remove','clear','solve','save'])
points = [{'marker': str(i), 'pixel': [i+1,i+1], 'reprojection_error_px': .1} for i in range(4)]
candidate = {'candidate_id':'candidate-1','saved':False,'points':points,'projections':[],
             'translation':[0,0,1],'quaternion_xyzw':[0,0,0,1],
             'mean_reprojection_error_px':.1,'max_reprojection_error_px':.1}
state = {'mode':'frozen','generation':1,'result':candidate,'parent_frame':'world','child_frame':'camera',
         'markers':[{'name':str(i)} for i in range(6)],'output_file':'',
         'source':{'image_ready':True,'intrinsic_ready':True,'marker_count':6,'image_topic':'image','intrinsic_file':'','intrinsic_source':'ideal-pinhole','ideal_horizontal_fov_degrees':110,'pose_prefix':'pose'},
         'frame':{'width':640,'height':480,'stamp_sec':1}}
with sync_playwright() as p:
    browser = p.chromium.launch(headless=True)
    try:
        for edit in ('add','remove','failed-solve','clear'):
            page = browser.new_page()
            failures = []
            page.on('pageerror', lambda error: failures.append(str(error)))
            saves = []
            fail_solve = [False]
            def route(req):
                path = req.request.url.split('/')[-1]
                if path == 'state': req.fulfill(json=state)
                elif path == 'solve':
                    req.fulfill(status=422 if fail_solve[0] else 200, json={'error':'fixture rejection'} if fail_solve[0] else candidate)
                elif path == 'save':
                    saves.append(req.request.post_data_json)
                    req.fulfill(json={**candidate,'saved':True,'output_file':'/saved.yaml'})
                elif 'image.jpg' in path:
                    req.fulfill(status=404)
                else: req.fulfill(content_type='text/html',body=html)
            page.route('http://fixture.test/**', route)
            page.goto('http://fixture.test/')
            # Production source uses no TS-only syntax; load exactly as a module.
            page.add_script_tag(type='module',content=source)
            page.wait_for_function("!document.getElementById('solve-button').disabled")
            assert "HFOV 110°" in page.locator("#intrinsic-file").inner_text()
            assert "assumed, not measured" in page.locator("#intrinsic-file").inner_text()
            page.click('#solve-button')
            page.wait_for_function("!document.getElementById('save-button').disabled")
            page.route('http://fixture.test/api/v1/image.jpg*', lambda r: r.fulfill(content_type='image/svg+xml',body='<svg xmlns="http://www.w3.org/2000/svg" width="640" height="480"/>'))
            page.wait_for_function("document.getElementById('camera-placeholder').classList.contains('hidden')")
            if edit == 'add':
                page.click('#camera-canvas',position={'x':20,'y':20})
            elif edit == 'remove': page.click('#remove-button')
            elif edit == 'clear': page.click('#clear-button')
            else:
                fail_solve[0]=True
                page.click('#solve-button')
                page.wait_for_function("document.getElementById('error-banner').textContent.includes('fixture rejection')")
            assert page.is_disabled('#save-button'), edit + ': stale Save remained enabled'
            assert not page.locator('#success-banner').inner_text(), edit + ': old success message retained'
            assert not saves, edit + ': edit dispatched Save'
            # A later successful replacement is the only path back to Save.
            fail_solve[0]=False
            for _ in range({'remove':1,'clear':4}.get(edit,0)):
                page.click('#camera-canvas',position={'x':30,'y':30})
            page.click('#solve-button')
            page.wait_for_function("!document.getElementById('save-button').disabled")
            page.click('#save-button')
            page.wait_for_function("document.getElementById('success-banner').textContent.includes('saved to')")
            assert saves == [{'candidate_id':'candidate-1'}], saves
            assert not failures, failures
            page.close()
            print(edit + ': passed')
    finally:
        browser.close()
