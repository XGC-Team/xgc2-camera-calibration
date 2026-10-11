/** Built pages render absent peers and recover over one SSE connection. */
import assert from 'node:assert/strict';
import fs from 'node:fs';
import path from 'node:path';
import {createRequire} from 'node:module';
const require=createRequire(process.env.PLAYWRIGHT_PACKAGE || import.meta.url);
const {chromium}=require('@playwright/test');
const root=path.resolve(import.meta.dirname, '../../xgc_camera_calibration/web');
const browser=await chromium.launch({headless:true, executablePath:process.env.CHROME_BIN || '/usr/bin/google-chrome'});
try {
  for(const kind of ['intrinsic', 'extrinsic']) {
    const page=await browser.newPage();
    const errors=[];
    let navigations=0, stateReads=0;
    page.on('pageerror',error=>errors.push(String(error)));
    page.on('framenavigated',frame=>{if(frame===page.mainFrame()) navigations++;});
    await page.addInitScript(()=>{
      window.peerEvents=[]; window.retryDelays=[];
      const timeout=window.setTimeout;
      window.setTimeout=(callback,delay,...args)=>{
        if(window.recordRetries) window.retryDelays.push(delay);
        return timeout(callback,delay,...args);
      };
      window.EventSource=class {
        constructor(url){this.url=url;this.listeners={};window.peerEvents.push(this);}
        addEventListener(name,callback){this.listeners[name]=callback;}
        close(){this.closed=true;}
      };
      window.peerState=value=>window.peerEvents.filter(peer=>!peer.closed)
        .forEach(peer=>peer.listeners.state({data:JSON.stringify(value)}));
    });
    const peers={ros_master:{available:false},media:{ready:false},camera_info:{received:false},world:{pose_available:false}};
    const state=kind==='intrinsic' ? {phase:'collecting',samples:0,coverage:[],candidate_pool:{count:0},
      image_ready:false,targets:[],detection:{status:'waiting'},peers} : {
      sampling_session_id:'a'.repeat(32),dataset_revision:0,samples:[],mode:'live',generation:0,result:null,
      parent_frame:'world',child_frame:'optical',frame:null,peers,
      source:{source_id:'camera',image_ready:false,intrinsic_ready:false,marker_count:0,marker_names:[],
        image_topic:'camera',pose_prefix:'pose',camera_info_received:false,world_pose_available:false}};
    await page.route('http://fixture.test/**',async route=>{
      const pathname=new URL(route.request().url()).pathname;
      if(pathname.endsWith('/state')){stateReads++;await route.fulfill({json:state});}
      else if(pathname.endsWith('/targets'))await route.fulfill({json:{board:null,views:[]}});
      else {
        const name=pathname.endsWith('/') ? 'index.html' : path.basename(pathname);
        await route.fulfill({path:path.join(root,kind,name),contentType:name.endsWith('.js') ? 'application/javascript'
          : name.endsWith('.css') ? 'text/css' : 'text/html'});
      }
    });
    await page.goto('http://fixture.test/'+kind+'/');
    await page.waitForFunction(()=>window.peerEvents.length===1);
    for(const id of ['ros-peer','media-peer','info-peer','world-peer'])
      assert.match(await page.locator('#'+id).textContent(), /waiting/i);
    // Peer status comes from the service stream, without a page navigation.
    peers.ros_master.available=peers.media.ready=peers.camera_info.received=peers.world.pose_available=true;
    if(kind==='extrinsic')state.source.camera_info_received=state.source.world_pose_available=true;
    await page.evaluate(value=>window.peerState(value),state);
    await page.waitForFunction(()=>document.getElementById('world-peer').textContent.includes('available'));
    assert.equal(await page.locator('#info-peer').textContent(), 'CameraInfo received');
    peers.media.ready=false;peers.world.pose_available=false;
    if(kind==='extrinsic')state.source.world_pose_available=false;
    await page.evaluate(value=>window.peerState(value),state);
    await page.waitForFunction(()=>document.getElementById('world-peer').textContent.includes('waiting'));
    // A close retires the sole event source before one jittered retry.
    await page.evaluate(()=>{
      window.recordRetries=true;
      window.peerEvents.at(-1).onerror();
      window.peerEvents.at(-1).onerror();
    });
    await page.waitForFunction(()=>window.peerEvents.length===2);
    await page.evaluate(()=>{window.recordRetries=false;});
    peers.media.ready=peers.world.pose_available=true;
    if(kind==='extrinsic')state.source.world_pose_available=true;
    await page.evaluate(value=>window.peerState(value),state);
    await page.waitForFunction(()=>document.getElementById('world-peer').textContent.includes('available'));
    const connections=await page.evaluate(()=>({active:window.peerEvents.filter(peer=>!peer.closed).length,
      delays:window.retryDelays}));
    assert.equal(connections.active,1);
    assert.equal(connections.delays.length,1);
    assert(connections.delays[0]>=0 && connections.delays[0]<=250);
    assert.equal(stateReads,1);
    assert.equal(navigations,1);
    assert.deepEqual(errors,[]);
    await page.evaluate(()=>window.dispatchEvent(new Event('pagehide')));
    assert.equal(await page.evaluate(()=>window.peerEvents.filter(peer=>!peer.closed).length),0);
    await page.close();
    console.log('PASS: '+kind+' renders immediately, streams per-peer recovery, retries once with jitter, and closes its stream');
  }
} finally {await browser.close();}
