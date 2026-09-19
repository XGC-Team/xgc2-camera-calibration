// @ts-nocheck -- imperative canvas and ROS transport retained behind the React view.
"use strict";

const ui = {
  canvas: document.getElementById("camera-canvas"),
  placeholder: document.getElementById("camera-placeholder"),
  modeChip: document.getElementById("mode-chip"),
  inputChip: document.getElementById("input-chip"),
  error: document.getElementById("error-banner"),
  success: document.getElementById("success-banner"),
  frameMeta: document.getElementById("frame-meta"),
  coordinateHint: document.getElementById("coordinate-hint"),
  markerSelect: document.getElementById("marker-select"),
  pointsBody: document.getElementById("points-body"),
  result: document.getElementById("result-box"),
  freeze: document.getElementById("freeze-button"),
  live: document.getElementById("live-button"),
  remove: document.getElementById("remove-button"),
  clear: document.getElementById("clear-button"),
  solve: document.getElementById("solve-button"),
  save: document.getElementById("save-button"),
  imageTopic: document.getElementById("image-topic"),
  intrinsicFile: document.getElementById("intrinsic-file"),
  posePrefix: document.getElementById("pose-prefix"),
  outputFile: document.getElementById("output-file"),
};

const context = ui.canvas.getContext("2d");
const state = { server:null, frameImage:null, display:null, points:[], candidate:null,
  busy:false, frameLoading:false, stateLoading:false, reviewId:null, replaceId:null,
  pending:null, disposed:false, controller:new AbortController(), frameEpoch:0 };
const pageEpoch = `page-${Date.now()}-${Math.random().toString(16).slice(2)}`;
let nextID = 0, loadVersion = 0, mutationEpoch = 0;
const uniqueID = () => `${pageEpoch}-${++nextID}`;

function showBanner(element,message) {
  element.textContent=message || "";
  element.classList.toggle("hidden",!message);
}
function clearMessages() { showBanner(ui.error,"");showBanner(ui.success,""); }
async function api(path,options={}) {
  const response=await fetch(path,{cache:"no-store",signal:state.controller.signal,...options});
  const payload=response.headers.get("content-type")?.includes("application/json") ? await response.json() : null;
  if (!response.ok) throw new Error(payload?.error || `Request failed (${response.status})`);
  return payload;
}
function post(path,body={}) {
  return api(path,{method:"POST",headers:{"Content-Type":"application/json"},body:JSON.stringify(body)});
}
function revision() {
  return {sampling_session_id:state.server.sampling_session_id,expected_revision:state.server.dataset_revision};
}
function setBusy(busy) { state.busy=busy;renderControls(); }
function renderState(server) {
  if (state.disposed) return;
  if (state.server?.sampling_session_id===server.sampling_session_id
      && server.dataset_revision<state.server.dataset_revision) return;
  if (state.server && state.server.sampling_session_id!==server.sampling_session_id) {
    state.frameImage=null;state.display=null;state.reviewId=null;state.replaceId=null;state.frameEpoch++;
  }
  state.server=server;state.points=server.samples || [];state.candidate=server.result;
  if (state.reviewId && !state.points.some(point=>point.sample_id===state.reviewId)) {
    state.reviewId=null;state.frameImage=null;state.display=null;
  }
  ui.modeChip.textContent=state.reviewId ? "Sample review" : server.mode==="frozen" ? "Paused" : "Live";
  ui.inputChip.textContent=`${server.source.marker_count} pose markers`;
  ui.imageTopic.textContent=server.source.image_topic;ui.posePrefix.textContent=server.source.pose_prefix;
  ui.outputFile.textContent=server.output_file || "—";
  const model=server.result?.camera_model || server.frame?.camera_model || server.source;
  ui.intrinsicFile.textContent=model.intrinsic_source==="ideal-pinhole"
    ? `Ideal pinhole · HFOV ${model.ideal_horizontal_fov_degrees}° · assumed, not measured`
    : model.intrinsic_file || "Model source unspecified";
  ui.coordinateHint.textContent=state.reviewId ? "Click to correct this sample on its original image; Resample uses a new observation."
    : state.replaceId ? "Click the selected rigid body to replace this sample. Its old observation is retained until success."
    : "Keep the camera fixed. Hold the selected rigid body still for each click; move it between clicks.";
  ui.result.textContent=state.candidate ? formatResult(state.candidate,server) : "Collect at least four independent correspondences.";
  renderControls();fitCanvas();
}
function renderControls() {
  const old=ui.markerSelect.value;
  ui.markerSelect.replaceChildren();
  for (const name of state.server?.source.marker_names || []) {
    const option=document.createElement("option");option.value=name;option.textContent=name;ui.markerSelect.appendChild(option);
  }
  if ([...ui.markerSelect.options].some(option=>option.value===old)) ui.markerSelect.value=old;
  ui.markerSelect.disabled=state.busy || !ui.markerSelect.options.length || !!state.reviewId;
  ui.freeze.disabled=state.busy || !state.server?.source.image_ready;
  ui.live.disabled=state.busy || (!state.reviewId && !state.replaceId && state.server?.mode==="live");
  ui.remove.disabled=state.busy || !state.points.length;ui.clear.disabled=state.busy || !state.points.length;
  ui.solve.disabled=state.busy || state.points.length<4;ui.save.disabled=state.busy || !state.candidate || state.candidate.saved;
  ui.pointsBody.replaceChildren();
  const errors=new Map((state.candidate?.points || []).map(point=>[point.sample_id,point]));
  for (const [index,point] of state.points.entries()) {
    const row=document.createElement("tr");row.dataset.sampleId=point.sample_id;
    const solved=errors.get(point.sample_id);
    for (const value of [`${index+1}. ${point.marker}`,point.pixel[0].toFixed(1),point.pixel[1].toFixed(1),solved?.reprojection_error_px?.toFixed(2) || "—"]) {
      const cell=document.createElement("td");cell.textContent=value;row.appendChild(cell);
    }
    const cell=document.createElement("td");
    for (const [label,action] of [["View",()=>review(point)], ["Resample",()=>resample(point)],
      ["Remove",()=>mutate("remove",{sample_id:point.sample_id})]]) {
      const button=document.createElement("button");button.textContent=label;button.type="button";button.disabled=state.busy;
      button.setAttribute("aria-label",`${label} sample ${index+1}`);button.addEventListener("click",action);cell.appendChild(button);
    }
    row.appendChild(cell);ui.pointsBody.appendChild(row);
  }
}
function fitCanvas() {
  const image=state.frameImage;if (!image) return;
  const bounds=ui.canvas.parentElement.getBoundingClientRect();
  const ratio=Math.min(bounds.width/image.naturalWidth,bounds.height/image.naturalHeight);
  const width=Math.max(1,Math.floor(image.naturalWidth*ratio)),height=Math.max(1,Math.floor(image.naturalHeight*ratio));
  const deviceRatio=window.devicePixelRatio || 1;
  ui.canvas.width=Math.floor(width*deviceRatio);ui.canvas.height=Math.floor(height*deviceRatio);
  ui.canvas.style.width=`${width}px`;ui.canvas.style.height=`${height}px`;
  context.setTransform(deviceRatio,0,0,deviceRatio,0,0);context.drawImage(image,0,0,width,height);
  ui.placeholder.classList.add("hidden");ui.frameMeta.textContent=`${image.naturalWidth}×${image.naturalHeight}`;
  if (state.reviewId) {
    const point=state.points.find(item=>item.sample_id===state.reviewId);
    const projection=state.candidate?.projections?.find(item=>item.sample_id===state.reviewId);
    context.font="12px ui-monospace,monospace";context.lineWidth=2;
    for (const [item,color] of [[point,"#ff5d67"],[projection,"#4ce691"]]) {
      if (!item) continue;
      const x=item.pixel[0]*width/image.naturalWidth,y=item.pixel[1]*height/image.naturalHeight;
      context.strokeStyle=color;context.beginPath();context.arc(x,y,6,0,Math.PI*2);context.stroke();
    }
  }
}
async function loadFrame(path="api/v1/image.jpg",sample=null,force=false) {
  if ((state.frameLoading && !force) || !state.server) return;
  const version=++loadVersion;
  const session=state.server.sampling_session_id, epoch=state.frameEpoch;
  state.frameLoading=true;let url;
  try {
    const response=await fetch(path,{cache:"no-store",signal:state.controller.signal});
    if (!response.ok) throw new Error(`Camera image unavailable (${response.status})`);
    url=URL.createObjectURL(await response.blob());const image=new Image();image.src=url;await image.decode();
    if (state.disposed || epoch!==state.frameEpoch || session!==state.server?.sampling_session_id) return;
    state.frameImage=image;
    state.display=sample?.display || {id:uniqueID(),source_id:state.server.source.source_id,source_epoch:pageEpoch,
      width:image.naturalWidth,height:image.naturalHeight,clock_domain:"browser-performance",
      presented_at_ms:performance.now(),time_origin_ms:performance.timeOrigin};
    fitCanvas();
  } catch (error) { if (!state.disposed) showBanner(ui.error,error.message); }
  finally { if (url) URL.revokeObjectURL(url);if(version===loadVersion) state.frameLoading=false; }
}
async function refreshState() {
  if (state.busy || state.stateLoading) return;
  state.stateLoading=true;const epoch=mutationEpoch;
  try { const server=await api("api/v1/state");if(epoch===mutationEpoch) renderState(server); }
  catch(error) { if (!state.disposed) showBanner(ui.error,error.message); }
  finally { state.stateLoading=false; }
}
async function action(operation) {
  if (state.busy || state.disposed) return;
  mutationEpoch++;clearMessages();setBusy(true);
  try { await operation(); }
  catch(error) { if (!state.disposed) { showBanner(ui.error,error.message);await refreshAfterFailure(); } }
  finally { setBusy(false); }
}
async function refreshAfterFailure() {
  try { renderState(await api("api/v1/state")); } catch(_) { /* original error remains visible */ }
}
async function freezeFrame() { await action(async()=>{
  state.frameEpoch++;state.reviewId=null;renderState(await post("api/v1/freeze"));await loadFrame("api/v1/image.jpg",null,true);
}); }
async function liveFrame() { await action(async()=>{
  state.frameEpoch++;state.reviewId=null;state.replaceId=null;renderState(await post("api/v1/live"));await loadFrame("api/v1/image.jpg",null,true);
}); }
async function review(point) { await action(async()=>{
  state.frameEpoch++;state.reviewId=point.sample_id;state.replaceId=null;
  state.frameImage=null;state.display=null;await loadFrame(point.image.path,point,true);renderState(state.server);
}); }
async function resample(point) { await action(async()=>{
  state.frameEpoch++;state.reviewId=null;state.replaceId=point.sample_id;
  renderState(await post("api/v1/live"));ui.markerSelect.value=point.marker;await loadFrame("api/v1/image.jpg",null,true);
}); }
async function mutate(operation,fields={}) { await action(async()=>{
  renderState(await post(`api/v1/samples/${operation}`,{...revision(),...fields}));
}); }
async function solve() { await action(async()=>{
  const result=await post("api/v1/solve",revision());renderState({...state.server,result});
  showBanner(ui.success,"Extrinsic candidate ready; save it explicitly after review.");
}); }
async function save() { if (!state.candidate || state.candidate.saved) return;await action(async()=>{
  const result=await post("api/v1/save",{candidate_id:state.candidate.candidate_id});
  renderState({...state.server,result,output_file:result.output_file});showBanner(ui.success,"Calibration saved.");
}); }
function formatResult(result,server) {
  return [`${server.parent_frame} → ${server.child_frame}`,
    `xyz [${result.translation.map(value=>value.toFixed(6)).join(", ")}]`,
    `q_xyzw [${result.quaternion_xyzw.map(value=>value.toFixed(6)).join(", ")}]`,
    `mean ${result.mean_reprojection_error_px.toFixed(3)} px, max ${result.max_reprojection_error_px.toFixed(3)} px`,
    ...(result.warnings || [])].join("\n");
}
ui.canvas.addEventListener("click",(event)=>{
  if (state.busy || !state.server || !state.frameImage || !state.display) return;
  const image=state.frameImage,bounds=ui.canvas.getBoundingClientRect();
  const pixel=[(event.clientX-bounds.left)*image.naturalWidth/bounds.width,(event.clientY-bounds.top)*image.naturalHeight/bounds.height];
  if (pixel[0]<0 || pixel[1]<0 || pixel[0]>=image.naturalWidth || pixel[1]>=image.naturalHeight) return;
  if (state.reviewId) { void mutate("pixel",{sample_id:state.reviewId,pixel});return; }
  const marker=ui.markerSelect.value;if (!marker) return;
  // Capture synchronously from the exact original Image used to paint the
  // visible canvas, not its scaled/annotated canvas or a later incoming frame.
  const copy=document.createElement("canvas");copy.width=image.naturalWidth;copy.height=image.naturalHeight;
  copy.getContext("2d").drawImage(image,0,0);
  const request={...revision(),request_id:uniqueID(),marker,pixel,display:{...state.display},
    ...(state.replaceId ? {replaces_sample_id:state.replaceId} : {})};
  void action(async()=>{
    const admitted=post("api/v1/samples/begin",request);
    // Attach rejection handling immediately while PNG encoding is pending.
    admitted.catch(()=>undefined);
    const blobPromise=new Promise(resolve=>copy.toBlob(resolve,"image/png"));
    try {
      const pending=await admitted;state.pending={session:request.sampling_session_id,id:pending.sample_id};
      const blob=await blobPromise;if (!blob) throw new Error("Could not preserve the displayed image");
      const result=await api(`api/v1/samples/${pending.sample_id}/image`,{method:"POST",headers:{"Content-Type":"image/png"},body:blob});
      if (request.sampling_session_id!==state.server?.sampling_session_id) return;
      state.replaceId=null;renderState(result);
    } finally {
      if (state.pending) {
        const pending=state.pending;state.pending=null;
        await post("api/v1/samples/cancel",{sampling_session_id:pending.session,sample_id:pending.id}).catch(()=>undefined);
      }
    }
  });
});
ui.freeze.addEventListener("click",freezeFrame);ui.live.addEventListener("click",liveFrame);
ui.remove.addEventListener("click",()=>{const point=state.points.at(-1);if(point) void mutate("remove",{sample_id:point.sample_id});});
ui.clear.addEventListener("click",()=>mutate("clear"));ui.solve.addEventListener("click",solve);ui.save.addEventListener("click",save);
window.addEventListener("resize",fitCanvas);
void refreshState();
const stateTimer=window.setInterval(refreshState,1000);
const liveTimer=window.setInterval(()=>{
  if(state.server?.mode==="live" && !state.reviewId && !state.busy) void loadFrame();
},500);
window.addEventListener("pagehide",()=>{
  state.disposed=true;state.frameEpoch++;state.controller.abort();clearInterval(stateTimer);clearInterval(liveTimer);
  if(state.pending) void fetch("api/v1/samples/cancel",{method:"POST",keepalive:true,
    headers:{"Content-Type":"application/json"},body:JSON.stringify({sampling_session_id:state.pending.session,sample_id:state.pending.id})}).catch(()=>undefined);
});
export {}
