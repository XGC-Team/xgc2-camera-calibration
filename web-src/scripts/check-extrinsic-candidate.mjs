/** Stateful API fixture exercises the real standalone consumer in Chromium. */
import assert from 'node:assert/strict';
import fs from 'node:fs';
import { createRequire } from 'node:module';
// Point PLAYWRIGHT_PACKAGE at an existing browser-test package manifest if
// Playwright is not installed alongside this standalone development checkout.
const require=createRequire(process.env.PLAYWRIGHT_PACKAGE||import.meta.url);
const {chromium}=require('@playwright/test');
const fixture=JSON.parse(fs.readFileSync(0,'utf8'));
const red=Buffer.from(fixture.red,'base64'),blue=Buffer.from(fixture.blue,'base64');
const state={sampling_session_id:'a'.repeat(32),dataset_revision:0,samples:[],mode:'live',generation:0,
 result:null,parent_frame:'world',child_frame:'camera',output_file:'',frame:null,
 source:{source_id:'camera',image_ready:true,intrinsic_ready:true,marker_count:1,marker_names:['one-body'],
 image_topic:'image',intrinsic_file:'',intrinsic_source:'ideal-pinhole',ideal_horizontal_fov_degrees:110,pose_prefix:'pose'}};
const pending=new Map(),images=new Map(),saves=[],requests=[],failures=[];
let identity=0,imageReads=0,heldImage,rejectCommit=false,resolveHeld;
const heldReady=new Promise(resolve=>{resolveHeld=resolve;});
let heldDeadline;
const browser=await chromium.launch({headless:true,executablePath:process.env.CHROME_BIN||'/usr/bin/google-chrome'});
try {
 const page=await browser.newPage({viewport:{width:1000,height:1000}});
 page.on('pageerror',error=>failures.push(String(error)));
 await page.route('http://fixture.test/**',async route=>{
  try {
   const request=route.request(),path=new URL(request.url()).pathname.slice(1);
   requests.push([request.method(),path]);
   if(request.method()==='GET') {
    if(path==='api/v1/state')await route.fulfill({json:state});
    else if(path==='api/v1/image.jpg'){imageReads++;if(imageReads===2){heldImage=route;resolveHeld();}else await route.fulfill({contentType:'image/png',body:imageReads===1?red:blue});}
    else if(images.has(path))await route.fulfill({contentType:'image/png',body:images.get(path)});
    else await route.fulfill({contentType:'text/html',body:fixture.html});
    return;
   }
   const body=request.headers()['content-type']==='application/json'?request.postDataJSON():null;
   if(path==='api/v1/samples/begin') {
    assert.equal(body.expected_revision,state.dataset_revision);assert.equal(body.sampling_session_id,state.sampling_session_id);
    assert.equal(body.marker,'one-body');const id=(++identity).toString(16).padStart(32,'0');pending.set(id,body);
    if(heldImage){await heldImage.fulfill({contentType:'image/png',body:blue});heldImage=undefined;}
    await route.fulfill({json:{sample_id:id,status:'pending',sampling_session_id:state.sampling_session_id,dataset_revision:state.dataset_revision}});
   } else if(path.startsWith('api/v1/samples/')&&path.endsWith('/image')) {
    if(rejectCommit){await route.fulfill({status:409,json:{error:'fixture commit rejected'}});return;}
    const id=path.split('/').at(-2),capture=pending.get(id);assert(capture);pending.delete(id);
    const sample={sample_id:id,marker:capture.marker,pixel:capture.pixel,world:[identity,0,1],display:capture.display,
     image:{path,mime_type:'image/png',width:640,height:480}};
    if(capture.replaces_sample_id)state.samples=state.samples.map(item=>item.sample_id===capture.replaces_sample_id?sample:item);else state.samples.push(sample);
    images.set(path,request.postDataBuffer());state.dataset_revision++;state.result=null;await route.fulfill({json:state});
   } else if(path==='api/v1/samples/cancel'){pending.delete(body.sample_id);await route.fulfill({json:state});}
   else if(['api/v1/samples/remove','api/v1/samples/clear','api/v1/samples/pixel'].includes(path)) {
    assert.equal(body.expected_revision,state.dataset_revision);
    if(path.endsWith('/clear'))state.samples=[];else if(path.endsWith('/remove'))state.samples=state.samples.filter(p=>p.sample_id!==body.sample_id);
    else state.samples.find(p=>p.sample_id===body.sample_id).pixel=body.pixel;
    state.dataset_revision++;state.result=null;await route.fulfill({json:state});
   } else if(path==='api/v1/solve') {
    assert.deepEqual(Object.keys(body).sort(),['expected_revision','sampling_session_id']);assert.equal(body.expected_revision,state.dataset_revision);
    state.result={candidate_id:'candidate-'+state.dataset_revision,saved:false,dataset_revision:state.dataset_revision,
     points:state.samples.map(p=>({...p,reprojection_error_px:.1})),projections:[],translation:[0,0,1],quaternion_xyzw:[0,0,0,1],
     mean_reprojection_error_px:.1,max_reprojection_error_px:.1};await route.fulfill({json:state.result});
   } else if(path==='api/v1/save') {
    assert.deepEqual(body,{candidate_id:state.result.candidate_id});saves.push(body);state.result={...state.result,saved:true,output_file:'/saved.yaml'};await route.fulfill({json:state.result});
   } else if(['api/v1/freeze','api/v1/live'].includes(path)){state.mode=path.endsWith('freeze')?'frozen':'live';await route.fulfill({json:state});}
   else throw Error('unexpected request '+path);
  } catch(error){failures.push(String(error));await route.fulfill({status:500,json:{error:String(error)}});}
 });
 await page.goto('http://fixture.test/');await page.addScriptTag({type:'module',content:fixture.source});
 await page.waitForFunction(()=>document.getElementById('camera-canvas').getContext('2d').getImageData(0,0,1,1).data[0]===255);
 // Actual second request is the boundary; no sleep-based readiness claim.
 try { await Promise.race([heldReady,new Promise((_,reject)=>{heldDeadline=setTimeout(()=>reject(Error('second preview request absent')),10000);})]); } finally {clearTimeout(heldDeadline);}
 assert(heldImage);
 await page.evaluate(()=>{const original=HTMLCanvasElement.prototype.toBlob;HTMLCanvasElement.prototype.toBlob=function(callback,...args){original.call(this,blob=>setTimeout(()=>callback(blob),150),...args);};});
 const ready=async count=>page.waitForFunction(n=>document.querySelectorAll('#points-body tr').length===n&&!document.getElementById('marker-select').disabled,count);
 await page.click('#camera-canvas',{position:{x:120,y:100}});await ready(1);
 const original=structuredClone(state.samples[0]);
 const color=await page.evaluate(async path=>{const image=new Image();image.src=path;await image.decode();const canvas=document.createElement('canvas');canvas.width=640;canvas.height=480;canvas.getContext('2d').drawImage(image,0,0);return [...canvas.getContext('2d').getImageData(0,0,1,1).data];},original.image.path);
 assert.deepEqual(color,[255,0,0,255]);assert.equal(await page.locator('#marker-select').inputValue(),'one-body');
 for(let i=0;i<3;i++){await page.click('#camera-canvas',{position:{x:200+i*20,y:150}});await ready(i+2);}
 assert.deepEqual(state.samples[0].world,original.world);assert.equal(new Set(state.samples.map(p=>p.sample_id)).size,4);
 assert(!requests.some(([,path])=>path==='api/v1/freeze'));
 await page.click('#solve-button');await page.waitForFunction(()=>!document.getElementById('save-button').disabled);
 const oldID=state.samples[0].sample_id,oldWorld=[...state.samples[0].world];
 await page.getByRole('button',{name:'View sample 1',exact:true}).click();
 await page.waitForFunction(()=>document.getElementById('coordinate-hint').textContent.includes('correct this sample'));
 await page.click('#camera-canvas',{position:{x:100,y:80}});
 await page.waitForFunction(()=>!document.getElementById('solve-button').disabled&&document.getElementById('save-button').disabled);
 assert.equal(state.samples[0].sample_id,oldID);assert.deepEqual(state.samples[0].world,oldWorld);assert.notDeepEqual(state.samples[0].pixel,original.pixel);
 rejectCommit=true;await page.getByRole('button',{name:'Resample sample 1',exact:true}).click();await ready(4);
 await page.click('#camera-canvas',{position:{x:180,y:120}});
 await page.waitForFunction(()=>document.getElementById('error-banner').textContent.includes('fixture commit rejected'));
 await ready(4);assert.equal(state.samples[0].sample_id,oldID);assert.equal(pending.size,0);
 rejectCommit=false;await page.click('#camera-canvas',{position:{x:180,y:120}});
 await page.waitForFunction(old=>document.querySelector('#points-body tr').dataset.sampleId!==old,oldID);await ready(4);
 await page.click('#solve-button');await page.waitForFunction(()=>!document.getElementById('save-button').disabled);
 await page.click('#save-button');await page.waitForFunction(()=>document.getElementById('success-banner').textContent.includes('saved'));
 assert.equal(saves.length,1);assert(await page.isDisabled('#save-button'));
 await page.getByRole('button',{name:'Remove sample 2',exact:true}).click();await ready(3);
 await page.click('#remove-button');await ready(2);await page.click('#clear-button');await ready(0);
 assert.deepEqual(failures,[]);
 console.log('PASS: live repeated marker; displayed A retained across late B; independent edit/resample/remove/undo/clear; exact candidate Save; no pageerror');
} finally {await browser.close();}
