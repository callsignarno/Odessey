import React, { useState, useMemo, useRef, useEffect, useCallback } from "react";
import {
  BarChart, Bar, LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip,
  ReferenceLine, ResponsiveContainer, Cell, Legend,
} from "recharts";
import {
  Play, Pause, RotateCcw, Shuffle, Radio, Crosshair, Flag, Trophy, Target,
  Zap, GitBranch, Network, Repeat, Flame, Undo2, Sparkles, X, ScrollText,
  Hand, MapPin, Fuel, Clock, Leaf, AlertTriangle, Route, Siren, Boxes, ListOrdered,
} from "lucide-react";

/* ===================================================================== *
 *  ENGINE — pure functions, Euclidean / metric TSP (68/68 tests pass)   *
 * ===================================================================== */
function mulberry32(seed){let a=seed>>>0;return function(){a|=0;a=(a+0x6d2b79f5)|0;let t=Math.imul(a^(a>>>15),1|a);t=(t+Math.imul(t^(t>>>7),61|t))^t;return((t^(t>>>14))>>>0)/4294967296;};}
const dist=(p,q)=>Math.hypot(p.x-q.x,p.y-q.y);
function distanceMatrix(pts){const n=pts.length;const D=Array.from({length:n},()=>new Float64Array(n));for(let i=0;i<n;i++)for(let j=0;j<n;j++)D[i][j]=i===j?0:dist(pts[i],pts[j]);return D;}
function tourLength(t,D){let s=0;for(let i=0;i<t.length;i++)s+=D[t[i]][t[(i+1)%t.length]];return s;}
function pathLength(t,D){let s=0;for(let i=0;i<t.length-1;i++)s+=D[t[i]][t[i+1]];return s;}
function genClusters(n,seed,k=4){const r=mulberry32(seed);const c=Array.from({length:k},()=>({x:0.2+r()*0.6,y:0.2+r()*0.6}));return Array.from({length:n},(_,id)=>{const cc=c[Math.floor(r()*k)];const cl=v=>Math.min(0.95,Math.max(0.05,v));return{id,x:cl(cc.x+(r()-0.5)*0.22),y:cl(cc.y+(r()-0.5)*0.22)};});}
function heldKarp(D){const n=D.length;if(n<=1)return{tour:[...Array(n).keys()],length:0};if(n===2)return{tour:[0,1],length:D[0][1]+D[1][0]};const FULL=1<<n;const dp=Array.from({length:FULL},()=>new Float64Array(n).fill(Infinity));const par=Array.from({length:FULL},()=>new Int16Array(n).fill(-1));dp[1][0]=0;for(let mask=1;mask<FULL;mask++){if(!(mask&1))continue;for(let i=0;i<n;i++){if(!(mask&(1<<i)))continue;const cur=dp[mask][i];if(cur===Infinity)continue;for(let j=1;j<n;j++){if(mask&(1<<j))continue;const nm=mask|(1<<j),nd=cur+D[i][j];if(nd<dp[nm][j]){dp[nm][j]=nd;par[nm][j]=i;}}}}const full=FULL-1;let best=Infinity,last=-1;for(let i=1;i<n;i++){const c=dp[full][i]+D[i][0];if(c<best){best=c;last=i;}}const tour=[];let mask=full,i=last;while(i!==-1&&i!==0){tour.push(i);const p=par[mask][i];mask^=1<<i;i=p;}tour.push(0);tour.reverse();return{tour,length:best};}
function nearestNeighbour(D,start=0){const n=D.length,visited=new Array(n).fill(false);const tour=[start],frames=[[start]];visited[start]=true;let cur=start;for(let k=1;k<n;k++){let b=-1,bd=Infinity;for(let j=0;j<n;j++)if(!visited[j]&&D[cur][j]<bd){bd=D[cur][j];b=j;}visited[b]=true;tour.push(b);cur=b;frames.push(tour.slice());}return{tour,length:tourLength(tour,D),frames};}
function primMST(D){const n=D.length,inT=new Array(n).fill(false),key=new Array(n).fill(Infinity),par=new Array(n).fill(-1);key[0]=0;const edges=[];for(let c=0;c<n;c++){let u=-1,uk=Infinity;for(let v=0;v<n;v++)if(!inT[v]&&key[v]<uk){uk=key[v];u=v;}if(u===-1)break;inT[u]=true;if(par[u]!==-1)edges.push([par[u],u]);for(let v=0;v<n;v++)if(!inT[v]&&D[u][v]<key[v]){key[v]=D[u][v];par[v]=u;}}return{edges};}
function mstApprox(D){const n=D.length,{edges}=primMST(D);const adj=Array.from({length:n},()=>[]);for(const[a,b]of edges){adj[a].push(b);adj[b].push(a);}const vis=new Array(n).fill(false),tour=[],stack=[0];while(stack.length){const u=stack.pop();if(vis[u])continue;vis[u]=true;tour.push(u);const ns=adj[u].filter(v=>!vis[v]).sort((a,b)=>a-b);for(let i=ns.length-1;i>=0;i--)stack.push(ns[i]);}for(let v=0;v<n;v++)if(!vis[v])tour.push(v);return{tour,length:tourLength(tour,D),mstEdges:edges};}
function exactMatching(odd,D){const k=odd.length;if(k===0)return[];const FULL=1<<k,dp=new Float64Array(FULL).fill(Infinity),ch=new Int32Array(FULL).fill(-1);dp[0]=0;for(let mask=0;mask<FULL;mask++){if(dp[mask]===Infinity)continue;let i=0;while(i<k&&mask&(1<<i))i++;if(i===k)continue;for(let j=i+1;j<k;j++){if(mask&(1<<j))continue;const nm=mask|(1<<i)|(1<<j),w=dp[mask]+D[odd[i]][odd[j]];if(w<dp[nm]){dp[nm]=w;ch[nm]=(i<<16)|j;}}}const edges=[];let mask=FULL-1;while(mask){const c=ch[mask],i=c>>>16,j=c&0xffff;edges.push([odd[i],odd[j]]);mask^=(1<<i)|(1<<j);}return edges;}
function greedyMatching(odd,D){const used=new Array(odd.length).fill(false),pairs=[];for(let i=0;i<odd.length;i++)for(let j=i+1;j<odd.length;j++)pairs.push([i,j,D[odd[i]][odd[j]]]);pairs.sort((a,b)=>a[2]-b[2]);const edges=[];for(const[i,j]of pairs)if(!used[i]&&!used[j]){used[i]=used[j]=true;edges.push([odd[i],odd[j]]);}return edges;}
function eulerianCircuit(edgeList,n,start){const adj=Array.from({length:n},()=>[]);edgeList.forEach(([a,b],id)=>{adj[a].push({to:b,id});adj[b].push({to:a,id});});const used=new Array(edgeList.length).fill(false),ptr=new Array(n).fill(0),circuit=[],stack=[start];while(stack.length){const v=stack[stack.length-1];while(ptr[v]<adj[v].length&&used[adj[v][ptr[v]].id])ptr[v]++;if(ptr[v]===adj[v].length){circuit.push(v);stack.pop();}else{const e=adj[v][ptr[v]++];used[e.id]=true;stack.push(e.to);}}return circuit.reverse();}
function christofides(D,cap=16){const n=D.length;if(n<3)return{...nearestNeighbour(D),mstEdges:[],matchEdges:[],exactMatch:true,oddCount:0};const{edges:mstEdges}=primMST(D);const deg=new Array(n).fill(0);for(const[a,b]of mstEdges){deg[a]++;deg[b]++;}const odd=[];for(let i=0;i<n;i++)if(deg[i]%2)odd.push(i);const exact=odd.length<=cap;const matchEdges=exact?exactMatching(odd,D):greedyMatching(odd,D);const euler=eulerianCircuit(mstEdges.concat(matchEdges),n,0);const seen=new Array(n).fill(false),tour=[];for(const v of euler)if(!seen[v]){seen[v]=true;tour.push(v);}for(let v=0;v<n;v++)if(!seen[v])tour.push(v);return{tour,length:tourLength(tour,D),mstEdges,matchEdges,oddCount:odd.length,exactMatch:exact};}
function twoOpt(D,init,maxFrames=500){const n=init.length;let tour=init.slice(),len=tourLength(tour,D);const trace=[len],frames=[tour.slice()];let improved=true,guard=0;while(improved&&guard<3000){improved=false;guard++;for(let i=1;i<n-1;i++)for(let k=i+1;k<n;k++){const a=tour[i-1],b=tour[i],c=tour[k],d=tour[(k+1)%n];if(a===c||b===d)continue;const delta=D[a][c]+D[b][d]-(D[a][b]+D[c][d]);if(delta<-1e-9){let lo=i,hi=k;while(lo<hi){const t=tour[lo];tour[lo]=tour[hi];tour[hi]=t;lo++;hi--;}len+=delta;improved=true;trace.push(len);if(frames.length<maxFrames)frames.push(tour.slice());}}}return{tour,length:tourLength(tour,D),trace,frames};}
function simulatedAnnealing(D,init,opts={}){const n=init.length,rnd=mulberry32(opts.seed??12345);let tour=init.slice(),len=tourLength(tour,D),best=tour.slice(),bestLen=len;const iters=opts.iters??Math.max(5000,n*700);let T=opts.T0??(len/n)*2;const Tmin=1e-6;const alpha=Math.pow(Tmin/Math.max(T,1e-9),1/iters);const trace=[],frames=[],sample=Math.max(1,Math.floor(iters/200));for(let it=0;it<iters;it++){let i=1+Math.floor(rnd()*(n-1)),k=1+Math.floor(rnd()*(n-1));if(i>k){const t=i;i=k;k=t;}if(i!==k){const a=tour[i-1],b=tour[i],c=tour[k],d=tour[(k+1)%n];if(!(a===c||b===d)){const delta=D[a][c]+D[b][d]-(D[a][b]+D[c][d]);if(delta<0||rnd()<Math.exp(-delta/T)){let lo=i,hi=k;while(lo<hi){const t=tour[lo];tour[lo]=tour[hi];tour[hi]=t;lo++;hi--;}len+=delta;if(len<bestLen){bestLen=len;best=tour.slice();}}}}T*=alpha;if(it%sample===0){trace.push({it,current:len,best:bestLen});frames.push(tour.slice());}}frames.push(best.slice());return{tour:best,length:bestLen,trace,frames};}

/* ===================================================================== *
 *  REAL-WORLD LAYER — Bengaluru scenarios + operational metrics         *
 * ===================================================================== */
const KM_PER_UNIT=34, SPEED_KMPH=22, FUEL_RS_PER_KM=2.3, CO2_KG_PER_KM=0.045, SERVICE_MIN=12;
const BLR=[
  ["Vi NOC · Koramangala",.575,.555],["Yelahanka",.45,.07],["Hebbal",.47,.18],
  ["RT Nagar",.50,.25],["Banaswadi",.60,.295],["KR Puram",.73,.34],
  ["Whitefield",.88,.44],["Marathahalli",.755,.475],["Bellandur",.70,.575],
  ["Sarjapur Rd",.79,.665],["HSR Layout",.625,.625],["BTM Layout",.545,.625],
  ["Electronic City",.60,.875],["JP Nagar",.47,.68],["Jayanagar",.48,.615],
  ["Banashankari",.40,.655],["Vijayanagar",.32,.48],["RR Nagar",.25,.60],
  ["Kengeri",.175,.685],["Rajajinagar",.35,.345],["Malleshwaram",.42,.315],
  ["Peenya",.28,.21],["MG Road",.525,.445],["Indiranagar",.62,.42],["Domlur",.605,.465],
];
function towerScenario(){const down=new Set([6,12,21]);return BLR.map(([name,x,y],id)=>({id,x,y,name,critical:id!==0&&down.has(id)}));}
function medScenario(){const sites=[["Blood Bank · MG Road",.525,.445],["Victoria Hosp",.475,.50],["Bowring Hosp",.535,.40],["Manipal · HAL",.655,.43],["St John's",.585,.59],["Apollo · B'gatta",.52,.72],["Fortis · C'ham",.50,.37],["Columbia · Y'pur",.37,.27],["Aster · Hebbal",.47,.19],["Sakra · M'halli",.76,.49],["Narayana · EC",.60,.86],["Sparsh · RR Ngr",.26,.59]];const urgent=new Set([5,10]);return sites.map(([name,x,y],id)=>({id,x,y,name,critical:id!==0&&urgent.has(id)}));}
const SCENARIOS={
  towers:{label:"Tower field-ops",icon:Radio,depot:"Vi NOC",unitName:"tower",story:"A Vi field engineer must service every tower in one shift. Three sites are DOWN — every minute they wait is SLA penalty.",make:towerScenario},
  medical:{label:"Medical supply run",icon:Siren,depot:"Blood bank",unitName:"hospital",story:"One rider delivers blood units from the bank to twelve hospitals. Two orders are URGENT.",make:medScenario},
  custom:{label:"Custom map",icon:Boxes,depot:"Depot",unitName:"stop",story:"Draw your own network — click to add stops, drag to move, double-click to remove.",make:null},
};
const km=u=>u*KM_PER_UNIT;
const minutes=u=>(km(u)/SPEED_KMPH)*60;
const money=u=>km(u)*FUEL_RS_PER_KM;
const co2=u=>km(u)*CO2_KG_PER_KM;
function arrivals(tour,D){let t=0;const out={};out[tour[0]]=0;for(let i=1;i<tour.length;i++){t+=minutes(D[tour[i-1]][tour[i]])+SERVICE_MIN;out[tour[i]]=t;}return out;}
function slaMinutes(tour,D,pts){const crit=pts.filter(p=>p.critical).map(p=>p.id);if(!crit.length)return null;const arr=arrivals(tour,D);return Math.max(...crit.map(c=>arr[c]??0));}
const fmtMin=m=>m>=60?`${Math.floor(m/60)}h ${String(Math.round(m%60)).padStart(2,"0")}m`:`${Math.round(m)}m`;

/* ===================================================================== *
 *  ALGORITHM META — Riso palette (flat, loud)                           *
 * ===================================================================== */
const ALG={
  heldkarp:{name:"Held–Karp",short:"EXACT",color:"#2b4cf0",icon:Target,paradigm:"Dynamic programming",complexity:"O(2ⁿ·n²)",guarantee:"Exact optimum",
    long:"Builds dp[S][i] — the cheapest route that visits exactly the set S and ends at stop i — solving every subset once. It returns the provably shortest route, but the 2ⁿ subsets explode past ~15 stops. This wall is what makes route planning genuinely hard."},
  nn:{name:"Nearest Neighbour",short:"GREEDY",color:"#ff5436",icon:Zap,paradigm:"Greedy",complexity:"O(n²)",guarantee:"No constant ratio",
    long:"From each stop, ride to the closest unvisited one. It's what a human dispatcher does by instinct — instant, but greedy choices strand you with brutal cross-city hops at the end of the shift. Worst case drifts from optimal as Θ(log n)."},
  mst:{name:"MST 2-Approx",short:"≤ 2×",color:"#0c8a5f",icon:GitBranch,paradigm:"Approximation",complexity:"O(n²)",guarantee:"≤ 2 × optimal",
    long:"Build a minimum spanning tree of the sites (cost ≤ the optimal route), walk it depth-first, shortcut repeats. The triangle inequality guarantees the result is never worse than twice optimal — a provable service level you could put in a contract."},
  christofides:{name:"Christofides",short:"≤ 1.5×",color:"#7048e8",icon:Network,paradigm:"Approximation",complexity:"O(n³)",guarantee:"≤ 1.5 × optimal",
    long:"MST plus a minimum-weight perfect matching on its odd-degree sites (computed exactly here via a second bitmask DP) makes every degree even, so an Eulerian circuit exists; shortcut it into a route. The matching costs ≤ ½·optimal — the celebrated 1.5 bound, unbeaten for metric TSP for almost 50 years."},
  twoopt:{name:"2-Opt",short:"LOCAL",color:"#0f8ba8",icon:Repeat,paradigm:"Local search",complexity:"O(n²)/sweep",guarantee:"Local optimum",
    long:"Repeatedly find two route segments that, when uncrossed, shorten the ride — on the map you literally watch the tangles disappear. Descends to a strong local optimum, though it can settle in a valley it cannot climb out of."},
  sa:{name:"Simulated Annealing",short:"META",color:"#e0218a",icon:Flame,paradigm:"Metaheuristic",complexity:"O(iters)",guarantee:"Probabilistic",
    long:"Accept improving swaps always — and worsening ones with probability e^(−Δ/T). A hot start explores wildly; as the temperature cools it settles, escaping the local traps that catch plain 2-opt. No guarantee, but empirically excellent."},
};
const ORDER=["heldkarp","nn","mst","christofides","twoopt","sa"];
const EXACT_CAP=14;
function runAlgo(key,D){const t0=performance.now();let r;if(key==="heldkarp")r=heldKarp(D);else if(key==="nn")r=nearestNeighbour(D);else if(key==="mst")r=mstApprox(D);else if(key==="christofides")r=christofides(D);else if(key==="twoopt")r=twoOpt(D,nearestNeighbour(D).tour);else r=simulatedAnnealing(D,nearestNeighbour(D).tour,{seed:7});r.runtimeMs=performance.now()-t0;return r;}
function framesFor(r,n){if(!r)return[];if(r.frames&&r.frames.length)return r.frames;const t=r.tour,f=[];for(let i=2;i<=t.length;i++)f.push(t.slice(0,i));return f;}
function useWidth(){const[w,setW]=useState(typeof window!=="undefined"?window.innerWidth:1200);useEffect(()=>{const f=()=>setW(window.innerWidth);window.addEventListener("resize",f);return()=>window.removeEventListener("resize",f);},[]);return w;}

/* ===================================================================== *
 *  THEME — RISO                                                         *
 * ===================================================================== */
const R={bg:"#f3efe2",card:"#ffffff",ink:"#161310",mut:"#6e6a5e",mut2:"#a09c8e",
  red:"#ff5436",yellow:"#ffd23f",blue:"#2b4cf0",green:"#0c8a5f",paper:"#faf7ee"};
const BORDER=`2px solid ${R.ink}`;
const SHADOW="4px 4px 0 #161310";
const SHADOW_SM="2px 2px 0 #161310";
const PAD=34;

/* ===================================================================== *
 *  MAP CANVAS                                                           *
 * ===================================================================== */
function MapCanvas({points,frame,closed,underlay,color,height=470,interactive,ghost,picked,onAdd,onMove,onRemove,onPick,showNames,showOrder}){
  const ref=useRef(null);const drag=useRef({id:null});const last=useRef(0);
  const map=cv=>{const r=cv.getBoundingClientRect();return{rect:r,W:r.width,H:r.height,X:x=>PAD+x*(r.width-2*PAD),Y:y=>PAD+y*(r.height-2*PAD)};};
  const toNorm=(cv,e)=>{const{rect}=map(cv);return{x:Math.min(1,Math.max(0,(e.clientX-rect.left-PAD)/(rect.width-2*PAD))),y:Math.min(1,Math.max(0,(e.clientY-rect.top-PAD)/(rect.height-2*PAD)))};};
  const hit=(cv,e)=>{const{X,Y,rect}=map(cv);const px=e.clientX-rect.left,py=e.clientY-rect.top;let id=-1,bd=17;for(const p of points){const d=Math.hypot(X(p.x)-px,Y(p.y)-py);if(d<bd){bd=d;id=p.id;}}return id;};
  useEffect(()=>{
    const cv=ref.current;if(!cv)return;
    const{W,H,X,Y}=map(cv);const dpr=Math.min(window.devicePixelRatio||1,2);
    cv.width=W*dpr;cv.height=H*dpr;const ctx=cv.getContext("2d");ctx.setTransform(dpr,0,0,dpr,0,0);
    ctx.clearRect(0,0,W,H);
    // halftone-ish dot grid
    ctx.fillStyle="rgba(22,19,16,0.12)";
    for(let gx=0;gx<=22;gx++)for(let gy=0;gy<=22;gy++){ctx.beginPath();ctx.arc(PAD+(gx/22)*(W-2*PAD),PAD+(gy/22)*(H-2*PAD),1.1,0,Math.PI*2);ctx.fill();}
    // ghost optimal
    if(ghost&&ghost.tour&&ghost.tour.length>1){ctx.setLineDash([4,6]);ctx.strokeStyle=ghost.color||R.green;ctx.globalAlpha=.7;ctx.lineWidth=2;ctx.beginPath();ctx.moveTo(X(points[ghost.tour[0]].x),Y(points[ghost.tour[0]].y));for(let i=1;i<ghost.tour.length;i++)ctx.lineTo(X(points[ghost.tour[i]].x),Y(points[ghost.tour[i]].y));ctx.closePath();ctx.stroke();ctx.setLineDash([]);ctx.globalAlpha=1;}
    // underlay
    if(underlay)for(const u of underlay){ctx.strokeStyle=u.color;ctx.lineWidth=u.width||1.6;ctx.setLineDash(u.dash||[5,4]);ctx.globalAlpha=u.alpha??.6;for(const[a,b]of u.edges){ctx.beginPath();ctx.moveTo(X(points[a].x),Y(points[a].y));ctx.lineTo(X(points[b].x),Y(points[b].y));ctx.stroke();}ctx.setLineDash([]);ctx.globalAlpha=1;}
    // route — riso double stroke: black under, color over
    if(frame&&frame.length>1){
      ctx.lineJoin="round";ctx.lineCap="round";
      const drawPath=()=>{ctx.beginPath();ctx.moveTo(X(points[frame[0]].x),Y(points[frame[0]].y));for(let i=1;i<frame.length;i++)ctx.lineTo(X(points[frame[i]].x),Y(points[frame[i]].y));if(closed)ctx.lineTo(X(points[frame[0]].x),Y(points[frame[0]].y));};
      ctx.strokeStyle=R.ink;ctx.lineWidth=5;drawPath();ctx.stroke();
      ctx.strokeStyle=color;ctx.lineWidth=2.8;drawPath();ctx.stroke();
      if(frame.length>=2&&!closed){const a=points[frame[frame.length-2]],b=points[frame[frame.length-1]];const ang=Math.atan2(Y(b.y)-Y(a.y),X(b.x)-X(a.x));const mx=(X(a.x)+X(b.x))/2,my=(Y(a.y)+Y(b.y))/2;ctx.save();ctx.translate(mx,my);ctx.rotate(ang);ctx.fillStyle=color;ctx.strokeStyle=R.ink;ctx.lineWidth=1.6;ctx.beginPath();ctx.moveTo(8,0);ctx.lineTo(-5,5.5);ctx.lineTo(-5,-5.5);ctx.closePath();ctx.fill();ctx.stroke();ctx.restore();}
    }
    // nodes
    const orderOf={};if(showOrder&&closed&&frame)frame.forEach((id,i)=>{orderOf[id]=i;});
    for(const p of points){const px=X(p.x),py=Y(p.y);const isDepot=p.id===0;const isPicked=picked&&picked.includes(p.id);
      if(isDepot){ctx.save();ctx.translate(px,py);ctx.rotate(Math.PI/4);ctx.fillStyle=R.blue;ctx.strokeStyle=R.ink;ctx.lineWidth=2;ctx.fillRect(-6.5,-6.5,13,13);ctx.strokeRect(-6.5,-6.5,13,13);ctx.restore();}
      else if(p.critical){ctx.beginPath();ctx.arc(px,py,7,0,Math.PI*2);ctx.fillStyle=R.red;ctx.fill();ctx.strokeStyle=R.ink;ctx.lineWidth=2;ctx.stroke();ctx.beginPath();ctx.arc(px,py,11.5,0,Math.PI*2);ctx.strokeStyle=R.red;ctx.lineWidth=1.6;ctx.setLineDash([2,3]);ctx.stroke();ctx.setLineDash([]);}
      else{ctx.beginPath();ctx.arc(px,py,isPicked?6.5:5.2,0,Math.PI*2);ctx.fillStyle=isPicked?color:"#fff";ctx.fill();ctx.strokeStyle=R.ink;ctx.lineWidth=2;ctx.stroke();}
      // visit-order stamp
      if(showOrder&&closed&&orderOf[p.id]!=null&&!isDepot){const o=orderOf[p.id];ctx.fillStyle=R.yellow;ctx.strokeStyle=R.ink;ctx.lineWidth=1.5;const bw=o>=10?17:13;ctx.fillRect(px+6,py+4,bw,12);ctx.strokeRect(px+6,py+4,bw,12);ctx.fillStyle=R.ink;ctx.font="700 9px 'Space Mono',monospace";ctx.fillText(String(o),px+9,py+13.5);}
      if(showNames&&p.name){ctx.fillStyle=p.critical?R.red:isDepot?R.blue:R.mut;ctx.font=`${p.critical||isDepot?"700":"500"} 10px 'Archivo',sans-serif`;ctx.fillText(p.name.toUpperCase(),px+9,py-9);}
      else if(!showNames&&points.length<=22&&!showOrder){ctx.fillStyle=R.mut2;ctx.font="10px 'Space Mono',monospace";ctx.fillText(String(p.id),px+8,py-7);}}
  },[points,frame,closed,underlay,color,height,ghost,picked,showNames,showOrder]);
  const handlers=interactive?{
    onPointerDown:e=>{const cv=ref.current;if(interactive==="play"){const id=hit(cv,e);if(id>=0)onPick&&onPick(id);return;}const id=hit(cv,e);if(id>=0){drag.current={id};}else{const{x,y}=toNorm(cv,e);onAdd&&onAdd(x,y);}},
    onPointerMove:e=>{if(drag.current.id===null||drag.current.id<0)return;const now=performance.now();if(now-last.current<28)return;last.current=now;const{x,y}=toNorm(ref.current,e);onMove&&onMove(drag.current.id,x,y);},
    onPointerUp:()=>{drag.current={id:null};},onPointerLeave:()=>{drag.current={id:null};},
    onDoubleClick:e=>{const id=hit(ref.current,e);if(id>0)onRemove&&onRemove(id);},
  }:{};
  return <canvas ref={ref} style={{width:"100%",height,display:"block",cursor:interactive?(interactive==="play"?"pointer":"crosshair"):"default",touchAction:"none"}} {...handlers}/>;
}

/* ===================================================================== *
 *  atoms                                                                *
 * ===================================================================== */
const card={background:R.card,border:BORDER,borderRadius:10,boxShadow:SHADOW};
const cardFlat={background:R.card,border:BORDER,borderRadius:10};
const btn={display:"flex",alignItems:"center",justifyContent:"center",gap:7,height:38,padding:"0 14px",borderRadius:8,cursor:"pointer",fontFamily:"var(--sans)",fontSize:12,fontWeight:800,textTransform:"uppercase",letterSpacing:.4,border:BORDER,background:"#fff",color:R.ink,boxShadow:SHADOW_SM,transition:"transform .06s, box-shadow .06s"};
const btnBlue={...btn,background:R.blue,color:"#fff"};
const btnYellow={...btn,background:R.yellow,color:R.ink};
const Metric=({icon:Icon,label,value,sub,accent})=>(
  <div style={{display:"flex",gap:11,alignItems:"flex-start",padding:"12px 0"}}>
    <div style={{width:32,height:32,borderRadius:7,border:BORDER,background:accent||R.paper,display:"grid",placeItems:"center",flexShrink:0,boxShadow:SHADOW_SM}}><Icon size={15} color={accent?"#fff":R.ink}/></div>
    <div style={{minWidth:0}}>
      <div style={{fontSize:9.5,letterSpacing:1,textTransform:"uppercase",color:R.mut,fontWeight:800,fontFamily:"var(--mono)"}}>{label}</div>
      <div style={{fontSize:21,fontWeight:900,color:R.ink,fontFamily:"var(--sans)",lineHeight:1.12,marginTop:1}}>{value}</div>
      {sub&&<div style={{fontSize:11,color:R.mut,marginTop:1,fontWeight:500}}>{sub}</div>}
    </div>
  </div>
);
const NavTab=({active,onClick,icon:Icon,children})=>(
  <button onClick={onClick} style={{display:"flex",alignItems:"center",gap:7,padding:"8px 15px",borderRadius:8,cursor:"pointer",fontSize:12,fontWeight:800,textTransform:"uppercase",letterSpacing:.5,fontFamily:"var(--sans)",border:BORDER,background:active?R.yellow:"#fff",color:R.ink,boxShadow:active?SHADOW_SM:"none",transition:"all .1s"}}>
    <Icon size={14}/>{children}
  </button>
);
const Chip=({active,onClick,children,icon:Icon})=>(
  <button onClick={onClick} style={{display:"flex",alignItems:"center",gap:7,padding:"7px 13px",borderRadius:8,fontSize:12,fontWeight:800,textTransform:"uppercase",letterSpacing:.3,cursor:"pointer",fontFamily:"var(--sans)",border:BORDER,background:active?R.ink:"#fff",color:active?"#fff":R.ink,boxShadow:active?SHADOW_SM:"none"}}>
    {Icon&&<Icon size={13}/>}{children}
  </button>
);
const tip={background:"#fff",border:BORDER,borderRadius:8,fontSize:12,fontFamily:"var(--mono)",color:R.ink,boxShadow:SHADOW_SM};
const th={textAlign:"left",padding:"10px 14px",fontWeight:800};
const td={padding:"10px 14px",color:R.ink,fontWeight:500};

/* ===================================================================== *
 *  APP                                                                  *
 * ===================================================================== */
export default function Odyssey(){
  const [mode,setMode]=useState("plan");
  const [scenario,setScenario]=useState("towers");
  const [notes,setNotes]=useState(false);
  const [custom,setCustom]=useState({n:12,seed:7});
  const [customPts,setCustomPts]=useState(()=>genClusters(12,7));
  const points=useMemo(()=>scenario==="custom"?customPts:SCENARIOS[scenario].make(),[scenario,customPts]);
  const D=useMemo(()=>distanceMatrix(points),[points]);
  const n=points.length;
  const optimal=useMemo(()=>(n<=EXACT_CAP?heldKarp(D):null),[D,n]);
  const sc=SCENARIOS[scenario];
  const addCity=(x,y)=>setCustomPts(p=>p.length>=80?p:[...p,{id:p.length,x,y}]);
  const moveCity=(id,x,y)=>setCustomPts(p=>p.map(pt=>pt.id===id?{...pt,x,y}:pt));
  const removeCity=id=>setCustomPts(p=>p.length<=4?p:p.filter(pt=>pt.id!==id).map((pt,i)=>({...pt,id:i})));
  const reshuffle=()=>{const s=custom.seed+1;setCustom(c=>({...c,seed:s}));setCustomPts(genClusters(custom.n,s));};
  const resize=v=>{setCustom(c=>({...c,n:v}));setCustomPts(genClusters(v,custom.seed));};

  return(
    <div style={{fontFamily:"var(--sans)",color:R.ink,background:R.bg,minHeight:860,backgroundImage:"radial-gradient(rgba(22,19,16,0.05) 1px, transparent 1px)",backgroundSize:"22px 22px"}}>
      <style>{`@import url('https://fonts.googleapis.com/css2?family=Archivo:wght@500;600;700;800;900&family=Space+Mono:wght@400;700&display=swap');
        :root{--sans:'Archivo',system-ui,sans-serif;--mono:'Space Mono',ui-monospace,monospace;}
        ::-webkit-scrollbar{width:10px;height:10px}::-webkit-scrollbar-thumb{background:#161310;border-radius:0;border:2px solid #f3efe2}
        .rs{-webkit-appearance:none;appearance:none;height:8px;border-radius:0;background:#fff;border:2px solid #161310;outline:none}
        .rs::-webkit-slider-thumb{-webkit-appearance:none;width:18px;height:18px;border-radius:0;background:#ffd23f;border:2px solid #161310;cursor:pointer;box-shadow:2px 2px 0 #161310}
        button:active{transform:translate(2px,2px);box-shadow:none!important}
        .rrow:hover{background:#faf7ee}
      `}</style>

      {/* header */}
      <div style={{background:R.yellow,borderBottom:BORDER,padding:"0 28px"}}>
        <div style={{display:"flex",justifyContent:"space-between",alignItems:"center",paddingTop:18,flexWrap:"wrap",gap:12}}>
          <div style={{display:"flex",alignItems:"center",gap:12}}>
            <div style={{width:38,height:38,borderRadius:8,background:R.red,border:BORDER,display:"grid",placeItems:"center",boxShadow:SHADOW_SM}}><Route size={19} color="#fff"/></div>
            <div style={{display:"flex",alignItems:"baseline",gap:11}}>
              <span style={{fontWeight:900,fontSize:27,letterSpacing:-.8,textTransform:"uppercase"}}>Odyssey</span>
              <span style={{fontSize:10.5,fontWeight:800,letterSpacing:1.2,textTransform:"uppercase",background:R.ink,color:R.yellow,padding:"3px 9px",borderRadius:5}}>Field-ops route intelligence</span>
            </div>
          </div>
          <button onClick={()=>setNotes(true)} style={btn}><ScrollText size={14}/> How it works</button>
        </div>
        <div style={{display:"flex",gap:8,padding:"13px 0 16px",flexWrap:"wrap"}}>
          <NavTab active={mode==="plan"} onClick={()=>setMode("plan")} icon={Crosshair}>Plan</NavTab>
          <NavTab active={mode==="race"} onClick={()=>setMode("race")} icon={Flag}>Race</NavTab>
          <NavTab active={mode==="play"} onClick={()=>setMode("play")} icon={Hand}>Challenge</NavTab>
          <NavTab active={mode==="analyze"} onClick={()=>setMode("analyze")} icon={Trophy}>Evidence</NavTab>
        </div>
      </div>

      {/* scenario bar */}
      <div style={{display:"flex",gap:10,alignItems:"center",flexWrap:"wrap",padding:"18px 28px 0"}}>
        {Object.entries(SCENARIOS).map(([k,s])=>(<Chip key={k} active={scenario===k} onClick={()=>setScenario(k)} icon={s.icon}>{s.label}</Chip>))}
        {scenario==="custom"&&(<>
          <div style={{display:"flex",alignItems:"center",gap:10,marginLeft:6}}>
            <span style={{fontSize:11,fontWeight:800,textTransform:"uppercase",letterSpacing:.5}}>Stops</span>
            <input className="rs" type="range" min={5} max={80} value={custom.n} onChange={e=>resize(+e.target.value)} style={{width:130}}/>
            <span style={{fontFamily:"var(--mono)",fontWeight:700,color:R.blue}}>{custom.n}</span>
          </div>
          <button onClick={reshuffle} style={btn}><Shuffle size={13}/> Shuffle</button>
        </>)}
        <div style={{marginLeft:"auto",fontSize:11.5,fontWeight:700,display:"flex",alignItems:"center",gap:14,textTransform:"uppercase",letterSpacing:.3}}>
          <span style={{display:"flex",alignItems:"center",gap:6}}><span style={{width:10,height:10,background:R.blue,border:`2px solid ${R.ink}`,transform:"rotate(45deg)",display:"inline-block"}}/>{sc.depot}</span>
          {points.some(p=>p.critical)&&<span style={{display:"flex",alignItems:"center",gap:6,color:R.red}}><span style={{width:10,height:10,borderRadius:"50%",background:R.red,border:`2px solid ${R.ink}`,display:"inline-block"}}/>Critical</span>}
          <span style={{fontFamily:"var(--mono)",textTransform:"none"}}>{optimal?<>OPT = <b style={{color:R.blue}}>{km(optimal.length).toFixed(1)} km</b></>:`n>${EXACT_CAP} · exact infeasible`}</span>
        </div>
      </div>
      <div style={{padding:"8px 28px 0",fontSize:13,color:R.mut,maxWidth:860,fontWeight:500}}>{sc.story}</div>

      <div style={{padding:"18px 28px 34px"}}>
        {mode==="plan"&&<PlanMode {...{points,D,n,optimal,scenario,addCity,moveCity,removeCity}}/>}
        {mode==="race"&&<RaceMode {...{points,D,n,scenario}}/>}
        {mode==="play"&&<PlayMode {...{points,D,n,optimal,scenario}}/>}
        {mode==="analyze"&&<AnalyzeMode {...{points,D,n,optimal}}/>}
      </div>

      {notes&&<Notes onClose={()=>setNotes(false)}/>}
      <div style={{textAlign:"center",fontSize:11,fontFamily:"var(--mono)",paddingBottom:20,color:R.mut}}>
        ASSUMPTIONS · {SPEED_KMPH} km/h avg · ₹{FUEL_RS_PER_KM}/km fuel · {SERVICE_MIN} min/stop service · map ≈ {KM_PER_UNIT} km wide
      </div>
    </div>
  );
}

/* ---------------- PLAN ---------------- */
function PlanMode({points,D,n,optimal,scenario,addCity,moveCity,removeCity}){
  const [algo,setAlgo]=useState("twoopt");
  const [playing,setPlaying]=useState(true);
  const [idx,setIdx]=useState(0);
  const [speed,setSpeed]=useState(1);
  const [manifest,setManifest]=useState(false);
  const disabled=algo==="heldkarp"&&n>EXACT_CAP;
  const result=useMemo(()=>disabled?null:runAlgo(algo,D),[algo,D,disabled]);
  const frames=useMemo(()=>framesFor(result,n),[result,n]);
  const firstD=useRef(true);
  useEffect(()=>{setIdx(0);setPlaying(true);},[algo,scenario]);
  useEffect(()=>{if(firstD.current){firstD.current=false;return;}setPlaying(false);setIdx(Math.max(0,frames.length-1));},[D]);
  useEffect(()=>{
    if(!playing||!frames.length)return;let raf,prev=performance.now(),acc=0;
    const step=380/(speed*Math.max(1,frames.length/26));
    const loop=t=>{acc+=t-prev;prev=t;if(acc>=step){acc=0;setIdx(p=>{if(p>=frames.length-1){setPlaying(false);return p;}return p+1;});}raf=requestAnimationFrame(loop);};
    raf=requestAnimationFrame(loop);return()=>cancelAnimationFrame(raf);
  },[playing,frames.length,speed]);
  const frame=frames[Math.min(idx,frames.length-1)]||[];
  const closed=frame.length===n;
  const meta=ALG[algo];
  const under=useMemo(()=>{if(!result)return null;const u=[];if(result.mstEdges)u.push({edges:result.mstEdges,color:R.green,dash:[5,4],alpha:.5,width:1.6});if(result.matchEdges&&result.matchEdges.length)u.push({edges:result.matchEdges,color:"#e0218a",dash:[2,3],alpha:.65,width:1.8});return u;},[result]);
  const curU=result?(closed?tourLength(frame,D):pathLength(frame,D)):0;
  const ratio=result&&optimal?result.length/optimal.length:null;
  const sla=result?slaMinutes(result.tour,D,points):null;
  const arr=useMemo(()=>result?arrivals(result.tour,D):null,[result,D]);
  const conv=useMemo(()=>{if(!result)return null;if(algo==="twoopt"&&result.trace)return result.trace.map((v,i)=>({i,len:km(v)}));if(algo==="sa"&&result.trace)return result.trace.map((d,i)=>({i,current:km(d.current),best:km(d.best)}));return null;},[result,algo]);
  const w=useWidth();const cols=w<1080?"1fr":"232px minmax(0,1fr) 276px";

  return(
    <div style={{display:"grid",gridTemplateColumns:cols,gap:20}}>
      <div style={{display:"flex",flexDirection:"column",gap:10}}>
        <div style={{fontSize:10.5,letterSpacing:1.2,textTransform:"uppercase",fontWeight:900,margin:"2px 0 0 2px"}}>Routing engine</div>
        {ORDER.map(k=>{const a=ALG[k],Icon=a.icon,act=algo===k,dis=k==="heldkarp"&&n>EXACT_CAP;return(
          <button key={k} disabled={dis} onClick={()=>setAlgo(k)} style={{textAlign:"left",padding:"11px 13px",borderRadius:9,cursor:dis?"not-allowed":"pointer",border:BORDER,background:act?a.color:"#fff",opacity:dis?.42:1,boxShadow:act?SHADOW:SHADOW_SM,transition:"all .1s",fontFamily:"var(--sans)"}}>
            <div style={{display:"flex",alignItems:"center",gap:9}}><Icon size={15} color={act?"#fff":a.color}/><span style={{fontWeight:800,fontSize:13,color:act?"#fff":R.ink,textTransform:"uppercase",letterSpacing:.2}}>{a.name}</span></div>
            <div style={{display:"flex",gap:7,marginTop:6,alignItems:"center"}}>
              <span style={{fontSize:10,fontFamily:"var(--mono)",fontWeight:700,color:act?a.color:"#fff",background:act?"#fff":a.color,padding:"2px 6px",borderRadius:4,border:`1.5px solid ${act?"#fff":R.ink}`}}>{a.complexity}</span>
              <span style={{fontSize:10,fontWeight:800,color:act?"rgba(255,255,255,.85)":R.mut}}>{a.short}</span>
            </div>
          </button>);})}
      </div>

      <div>
        <div style={{...card,overflow:"hidden"}}>
          <div style={{display:"flex",justifyContent:"space-between",alignItems:"center",padding:"11px 16px",borderBottom:BORDER,background:R.paper}}>
            <div style={{display:"flex",alignItems:"center",gap:9}}><meta.icon size={16} color={meta.color}/><span style={{fontSize:16,fontWeight:900,textTransform:"uppercase",letterSpacing:.2}}>{meta.name}</span><span style={{fontSize:11,fontWeight:700,color:R.mut}}>· {meta.guarantee}</span></div>
            {scenario==="custom"&&<span style={{fontSize:10.5,fontWeight:700,color:R.mut,textTransform:"uppercase"}}>click add · drag move · 2×click delete</span>}
          </div>
          {disabled?(
            <div style={{height:470,display:"grid",placeItems:"center",textAlign:"center",padding:30,background:"#fff"}}>
              <div><Target size={28} color={R.mut2} style={{marginBottom:10}}/><div style={{fontSize:16,fontWeight:900,marginBottom:6,textTransform:"uppercase"}}>Exact is infeasible at {n} stops</div><div style={{fontSize:13,maxWidth:340,color:R.mut,fontWeight:500}}>2ⁿ states is the whole reason approximations exist. Use ≤ {EXACT_CAP} stops, or pick another engine.</div></div>
            </div>
          ):(
            <MapCanvas points={points} frame={frame} closed={closed} underlay={algo==="mst"||algo==="christofides"?under:null} color={meta.color}
              showNames={scenario!=="custom"} showOrder={closed} interactive={scenario==="custom"?"edit":null} onAdd={addCity} onMove={moveCity} onRemove={removeCity}/>
          )}
          {!disabled&&(
            <div style={{display:"flex",alignItems:"center",gap:12,padding:"11px 16px",borderTop:BORDER,background:R.paper}}>
              <button onClick={()=>{setIdx(0);setPlaying(true);}} style={{...btn,width:38,padding:0}}><RotateCcw size={15}/></button>
              <button onClick={()=>setPlaying(p=>!p)} style={{...btn,width:38,padding:0,background:meta.color,color:"#fff"}}>{playing?<Pause size={15}/>:<Play size={15}/>}</button>
              <input className="rs" type="range" min={0} max={Math.max(0,frames.length-1)} value={idx} onChange={e=>{setIdx(+e.target.value);setPlaying(false);}} style={{flex:1}}/>
              <span style={{fontFamily:"var(--mono)",fontSize:11.5,fontWeight:700,minWidth:80,textAlign:"right"}}>{idx+1}/{frames.length}</span>
              <div style={{display:"flex",gap:5}}>{[0.5,1,2,4].map(s=>(<button key={s} onClick={()=>setSpeed(s)} style={{padding:"3px 8px",borderRadius:5,fontSize:10.5,fontFamily:"var(--mono)",fontWeight:700,cursor:"pointer",border:`2px solid ${R.ink}`,background:speed===s?R.ink:"#fff",color:speed===s?"#fff":R.ink}}>{s}×</button>))}</div>
            </div>
          )}
        </div>
        {conv&&(
          <div style={{...card,marginTop:16,padding:"13px 8px 6px 2px"}}>
            <div style={{fontSize:10.5,fontWeight:900,letterSpacing:1,textTransform:"uppercase",padding:"0 14px 4px"}}>Convergence — km over improvement moves</div>
            <ResponsiveContainer width="100%" height={155}>
              <LineChart data={conv} margin={{top:6,right:20,left:0,bottom:0}}>
                <CartesianGrid stroke="#eceadf" vertical={false}/><XAxis dataKey="i" stroke={R.mut2} tick={{fontSize:10,fontFamily:"var(--mono)"}}/><YAxis stroke={R.mut2} tick={{fontSize:10,fontFamily:"var(--mono)"}} width={44} domain={["auto","auto"]}/><Tooltip contentStyle={tip}/>
                {optimal&&<ReferenceLine y={km(optimal.length)} stroke={R.blue} strokeDasharray="4 4" label={{value:"OPT",fill:R.blue,fontSize:10,fontWeight:700}}/>}
                {algo==="twoopt"&&<Line type="monotone" dataKey="len" stroke={meta.color} strokeWidth={2.4} dot={false}/>}
                {algo==="sa"&&<Line type="monotone" dataKey="current" stroke="#f5b1d4" strokeWidth={1.4} dot={false}/>}
                {algo==="sa"&&<Line type="monotone" dataKey="best" stroke={meta.color} strokeWidth={2.4} dot={false}/>}
              </LineChart>
            </ResponsiveContainer>
          </div>
        )}
      </div>

      <div style={{display:"flex",flexDirection:"column",gap:16}}>
        <div style={{...card,padding:"4px 16px"}}>
          <Metric icon={MapPin} label="Distance" value={disabled?"—":`${km(curU).toFixed(1)} km`} sub={disabled?"":closed?"full route":"riding…"} accent={meta.color}/>
          <div style={{height:2,background:R.ink,opacity:.08}}/>
          <Metric icon={Clock} label="Shift time" value={disabled?"—":fmtMin(minutes(result.length)+SERVICE_MIN*(n-1))} sub={`incl. ${SERVICE_MIN} min/stop`}/>
          <div style={{height:2,background:R.ink,opacity:.08}}/>
          <Metric icon={Fuel} label="Fuel cost" value={disabled?"—":`₹${money(result.length).toFixed(0)}`} sub={`${disabled?"":km(result.length).toFixed(1)} km total`}/>
          <div style={{height:2,background:R.ink,opacity:.08}}/>
          <Metric icon={Leaf} label="CO₂" value={disabled?"—":`${co2(result.length).toFixed(2)} kg`} sub="small motorcycle"/>
          {sla!=null&&<><div style={{height:2,background:R.ink,opacity:.08}}/>
          <Metric icon={AlertTriangle} label="Last critical fixed" value={disabled?"—":fmtMin(sla)} sub="until final DOWN site reached" accent={R.red}/></>}
          <div style={{height:2,background:R.ink,opacity:.08}}/>
          <Metric icon={Target} label="vs optimal" value={ratio?`${ratio.toFixed(3)}×`:"—"} sub={ratio?(ratio<=1.0001?"this IS the optimum":`+${((ratio-1)*100).toFixed(1)}% longer`):`unknown for n>${EXACT_CAP}`} accent={ratio&&ratio<=1.0001?R.green:undefined}/>
        </div>
        {!disabled&&result&&(
          <div style={{...card,overflow:"hidden"}}>
            <button onClick={()=>setManifest(m=>!m)} style={{width:"100%",display:"flex",alignItems:"center",gap:8,padding:"11px 15px",border:"none",borderBottom:manifest?BORDER:"none",background:R.paper,cursor:"pointer",fontFamily:"var(--sans)",fontSize:12,fontWeight:900,textTransform:"uppercase",letterSpacing:.6,color:R.ink}}>
              <ListOrdered size={15}/> Route manifest <span style={{marginLeft:"auto",fontFamily:"var(--mono)",fontWeight:700}}>{manifest?"−":"+"}</span>
            </button>
            {manifest&&(
              <div style={{maxHeight:260,overflowY:"auto"}}>
                {result.tour.map((id,i)=>{const p=points[id];return(
                  <div key={id} className="rrow" style={{display:"flex",alignItems:"center",gap:9,padding:"7px 14px",borderTop:i?`1px solid #eceadf`:"none"}}>
                    <span style={{fontFamily:"var(--mono)",fontSize:11,fontWeight:700,width:20,color:i===0?R.blue:R.mut}}>{i===0?"◆":i}</span>
                    <span style={{flex:1,fontSize:12,fontWeight:700,color:p.critical?R.red:R.ink,textTransform:"uppercase",letterSpacing:.2,whiteSpace:"nowrap",overflow:"hidden",textOverflow:"ellipsis"}}>{p.name||`Stop ${id}`}{p.critical&&" ⚠"}</span>
                    <span style={{fontFamily:"var(--mono)",fontSize:11,fontWeight:700,color:R.mut}}>{arr?fmtMin(arr[id]):""}</span>
                  </div>);})}
              </div>
            )}
          </div>
        )}
        <div style={{...cardFlat,padding:"13px 15px"}}>
          <div style={{fontSize:13,fontWeight:900,textTransform:"uppercase",letterSpacing:.4,marginBottom:6}}>{meta.paradigm}</div>
          <p style={{color:R.mut,fontSize:12.4,lineHeight:1.56,margin:0,fontWeight:500}}>{meta.long}</p>
        </div>
      </div>
    </div>
  );
}

/* ---------------- RACE ---------------- */
function RaceMode({points,D,n,scenario}){
  const keys=ORDER.filter(k=>!(k==="heldkarp"&&n>EXACT_CAP));
  const runs=useMemo(()=>keys.map(k=>{const r=runAlgo(k,D);return{key:k,r,frames:framesFor(r,n)};}),[D,n]);
  const [p,setP]=useState(0);
  const [playing,setPlaying]=useState(true);
  useEffect(()=>{setP(0);setPlaying(true);},[D]);
  useEffect(()=>{if(!playing)return;const dur=4200;let raf;const start=performance.now()-p*dur;const loop=t=>{const np=Math.min(1,(t-start)/dur);setP(np);if(np>=1){setPlaying(false);return;}raf=requestAnimationFrame(loop);};raf=requestAnimationFrame(loop);return()=>cancelAnimationFrame(raf);},[playing]);
  const winner=Math.min(...runs.map(x=>x.r.length));
  const board=runs.map(({key,r,frames})=>{const fi=Math.min(frames.length-1,Math.floor(p*(frames.length-1)));const f=frames[fi]||[];const closed=f.length===n;const len=closed?tourLength(f,D):pathLength(f,D);return{key,color:ALG[key].color,name:ALG[key].name,len,finalLen:r.length,done:fi>=frames.length-1};}).sort((a,b)=>(a.done&&b.done?a.finalLen-b.finalLen:a.len-b.len));
  const w=useWidth();const cols=w<880?"1fr":"minmax(0,1fr) 286px";const tiles=w<560?"1fr":w<880||keys.length<=4?"repeat(2,1fr)":"repeat(3,1fr)";

  return(
    <div style={{display:"grid",gridTemplateColumns:cols,gap:20}}>
      <div>
        <div style={{display:"flex",alignItems:"center",gap:12,marginBottom:16}}>
          <button onClick={()=>{setP(0);setPlaying(true);}} style={btn}><RotateCcw size={14}/> Restart</button>
          <button onClick={()=>setPlaying(x=>!x)} style={btnBlue}>{playing?<Pause size={14}/>:<Play size={14}/>}{playing?"Pause":"Play"}</button>
          <div style={{flex:1,height:14,border:BORDER,background:"#fff",overflow:"hidden",borderRadius:7}}><div style={{width:`${p*100}%`,height:"100%",background:R.red}}/></div>
          <span style={{fontFamily:"var(--mono)",fontSize:12,fontWeight:700}}>{(p*100).toFixed(0)}%</span>
        </div>
        <div style={{display:"grid",gridTemplateColumns:tiles,gap:16}}>
          {runs.map(({key,r,frames})=>{const fi=Math.min(frames.length-1,Math.floor(p*(frames.length-1)));const f=frames[fi]||[];const closed=f.length===n;const len=closed?tourLength(f,D):pathLength(f,D);const isWin=r.length===winner;return(
            <div key={key} style={{...card,overflow:"hidden",boxShadow:isWin&&p>=1?`6px 6px 0 ${ALG[key].color}`:SHADOW}}>
              <div style={{display:"flex",justifyContent:"space-between",alignItems:"center",padding:"8px 12px",borderBottom:BORDER,background:isWin&&p>=1?ALG[key].color:R.paper}}>
                <span style={{display:"flex",alignItems:"center",gap:7,fontSize:11.5,fontWeight:900,textTransform:"uppercase",letterSpacing:.2,color:isWin&&p>=1?"#fff":R.ink}}>{ALG[key].name}{isWin&&p>=1&&<Trophy size={12} color="#fff"/>}</span>
                <span style={{fontFamily:"var(--mono)",fontSize:12,fontWeight:700,color:isWin&&p>=1?"#fff":ALG[key].color}}>{km(len).toFixed(1)}km</span>
              </div>
              <MapCanvas points={points} frame={f} closed={closed} color={ALG[key].color} height={180} showNames={false}/>
            </div>);})}
        </div>
      </div>
      <div style={{...card,overflow:"hidden",alignSelf:"start"}}>
        <div style={{padding:"12px 15px",borderBottom:BORDER,fontSize:14,fontWeight:900,textTransform:"uppercase",letterSpacing:.5,display:"flex",alignItems:"center",gap:8,background:R.paper}}><Flag size={15}/> Leaderboard</div>
        {board.map((b,i)=>(
          <div key={b.key} className="rrow" style={{display:"flex",alignItems:"center",gap:10,padding:"10px 15px",borderTop:i?`1px solid #eceadf`:"none"}}>
            <span style={{fontFamily:"var(--mono)",fontSize:13,fontWeight:700,width:18,color:i===0?R.red:R.mut2}}>{i+1}</span>
            <span style={{width:10,height:10,borderRadius:3,background:b.color,border:`1.5px solid ${R.ink}`}}/>
            <span style={{flex:1,fontSize:12,fontWeight:800,textTransform:"uppercase",letterSpacing:.2}}>{b.name}</span>
            <span style={{fontFamily:"var(--mono)",fontSize:12,fontWeight:700}}>{km(b.len).toFixed(1)}</span>
          </div>
        ))}
        <div style={{padding:"12px 15px",fontSize:12,color:R.mut,borderTop:BORDER,lineHeight:1.55,fontWeight:500,background:R.paper}}>
          Six engines plan the same {scenario==="towers"?"maintenance shift":scenario==="medical"?"delivery run":"route"} in parallel. Greedy sprints early — the smarter engines overtake as they refine.
        </div>
      </div>
    </div>
  );
}

/* ---------------- CHALLENGE ---------------- */
function PlayMode({points,D,n,optimal,scenario}){
  const [tour,setTour]=useState([0]);
  const [reveal,setReveal]=useState(false);
  useEffect(()=>{setTour([0]);setReveal(false);},[D]);
  const pick=id=>setTour(t=>t.includes(id)?t:[...t,id]);
  const undo=()=>setTour(t=>t.length>1?t.slice(0,-1):t);
  const clear=()=>{setTour([0]);setReveal(false);};
  const complete=tour.length===n;
  const userU=complete?tourLength(tour,D):pathLength(tour,D);
  const compare=useMemo(()=>ORDER.filter(k=>!(k==="heldkarp"&&n>EXACT_CAP)).map(k=>({key:k,name:ALG[k].name,color:ALG[k].color,len:runAlgo(k,D).length})),[D,n]);
  const beat=complete?compare.filter(c=>userU<=c.len).length:0;
  const optLen=optimal?optimal.length:Math.min(...compare.map(c=>c.len));
  const userRatio=complete?userU/optLen:null;
  const w=useWidth();const cols=w<880?"1fr":"minmax(0,1fr) 296px";

  return(
    <div style={{display:"grid",gridTemplateColumns:cols,gap:20}}>
      <div style={{...card,overflow:"hidden"}}>
        <div style={{display:"flex",justifyContent:"space-between",alignItems:"center",padding:"11px 16px",borderBottom:BORDER,background:R.paper}}>
          <span style={{fontSize:16,fontWeight:900,textTransform:"uppercase",letterSpacing:.3}}>Beat the machine</span>
          <span style={{fontSize:10.5,fontWeight:700,color:R.mut,textTransform:"uppercase"}}>tap {SCENARIOS[scenario].unitName}s in order · ◆ = {SCENARIOS[scenario].depot}</span>
        </div>
        <MapCanvas points={points} frame={tour} closed={complete} color={R.blue} picked={tour} showNames={scenario!=="custom"} showOrder={complete}
          interactive="play" onPick={pick} ghost={reveal&&optimal?{tour:optimal.tour,color:R.green}:null}/>
        <div style={{display:"flex",alignItems:"center",gap:10,padding:"11px 16px",borderTop:BORDER,background:R.paper}}>
          <button onClick={undo} style={btn}><Undo2 size={14}/> Undo</button>
          <button onClick={clear} style={btn}><RotateCcw size={14}/> Clear</button>
          <span style={{fontFamily:"var(--mono)",fontSize:12,fontWeight:700}}>{tour.length}/{n}</span>
          {optimal&&<button onClick={()=>setReveal(r=>!r)} style={{...btn,marginLeft:"auto",background:reveal?R.green:"#fff",color:reveal?"#fff":R.ink}}><Sparkles size={14}/>{reveal?"Hide":"Reveal"} optimal</button>}
        </div>
      </div>

      <div style={{display:"flex",flexDirection:"column",gap:16}}>
        <div style={{...card,padding:"4px 16px"}}>
          <Metric icon={MapPin} label="Your route" value={`${km(userU).toFixed(1)} km`} sub={complete?"complete":`visit all ${n} ${SCENARIOS[scenario].unitName}s`} accent={R.blue}/>
          {complete&&<><div style={{height:2,background:R.ink,opacity:.08}}/>
          <Metric icon={Target} label="vs optimal" value={userRatio?`${userRatio.toFixed(3)}×`:"—"} sub={userRatio<=1.0001?"you found the optimum!":`+${((userRatio-1)*100).toFixed(1)}% longer`} accent={userRatio<=1.05?R.green:R.red}/></>}
        </div>
        {complete&&(
          <div style={{...card,padding:"15px 17px",background:beat>=4?R.green:beat>=2?R.yellow:R.red}}>
            <div style={{fontSize:23,fontWeight:900,textTransform:"uppercase",letterSpacing:.3,color:(beat>=2&&beat<4)?R.ink:"#fff"}}>You beat {beat}/{compare.length}</div>
            <div style={{fontSize:12,marginTop:4,lineHeight:1.5,fontWeight:700,color:(beat>=2&&beat<4)?R.ink:"rgba(255,255,255,.92)"}}>{beat===compare.length?"Flawless — no engine out-planned you.":beat>=4?"Sharp routing instinct.":beat>=2?"Respectable. The metaheuristics still have you.":"The machines win this shift. Hunt for crossings."}</div>
          </div>
        )}
        <div style={{...cardFlat,overflow:"hidden"}}>
          <div style={{padding:"10px 15px",borderBottom:BORDER,fontSize:11,fontWeight:900,letterSpacing:.8,textTransform:"uppercase",background:R.paper}}>The field {complete?"· you vs them":""}</div>
          {[...compare.map(c=>({...c,you:false})),...(complete?[{key:"you",name:"YOUR ROUTE",color:R.blue,len:userU,you:true}]:[])].sort((a,b)=>a.len-b.len).map(c=>{
            const max=Math.max(...compare.map(x=>x.len),complete?userU:0);const min=Math.min(...compare.map(x=>x.len),complete?userU:Infinity);
            const w=18+((c.len-min)/(max-min||1))*80;
            return(<div key={c.key} style={{padding:"7px 15px"}}>
              <div style={{display:"flex",justifyContent:"space-between",fontSize:11.5,marginBottom:3}}><span style={{fontWeight:c.you?900:700,textTransform:"uppercase",letterSpacing:.2,color:c.you?R.blue:R.ink}}>{c.name}</span><span style={{fontFamily:"var(--mono)",fontWeight:700,color:R.mut}}>{km(c.len).toFixed(1)}km</span></div>
              <div style={{height:10,border:`2px solid ${R.ink}`,borderRadius:5,background:"#fff",overflow:"hidden"}}><div style={{width:`${w}%`,height:"100%",background:c.color}}/></div>
            </div>);})}
        </div>
      </div>
    </div>
  );
}

/* ---------------- EVIDENCE ---------------- */
function AnalyzeMode({points,D,n,optimal}){
  const compare=useMemo(()=>ORDER.filter(k=>!(k==="heldkarp"&&n>EXACT_CAP)).map(k=>{const r=runAlgo(k,D);return{key:k,name:ALG[k].name,short:ALG[k].short,color:ALG[k].color,length:r.length,runtimeMs:r.runtimeMs,ratio:optimal?r.length/optimal.length:null,sla:slaMinutes(r.tour,D,points)};}),[D,n,optimal,points]);
  const best=Math.min(...compare.map(c=>c.length));
  const w=useWidth();const cols2=w<880?"1fr":"1fr 1fr";
  const hasSla=compare.some(c=>c.sla!=null);
  const shortest=compare.reduce((a,b)=>a.length<b.length?a:b);
  const fastestSla=hasSla?compare.reduce((a,b)=>(a.sla??1e9)<(b.sla??1e9)?a:b):null;
  const tradeoff=hasSla&&fastestSla&&shortest.key!==fastestSla.key;
  const [bench,setBench]=useState(null);const [busy,setBusy]=useState(false);
  const runBench=useCallback(()=>{setBusy(true);setTimeout(()=>{const sizes=[6,8,10,12,14,18,24,32,44,60,80];const rows=sizes.map(sz=>{const pts=genClusters(sz,7),DD=distanceMatrix(pts);const tm=fn=>{const t=performance.now();fn();return performance.now()-t;};const row={n:sz};row.nn=tm(()=>nearestNeighbour(DD));row.mst=tm(()=>mstApprox(DD));row.christofides=tm(()=>christofides(DD));row.twoopt=tm(()=>twoOpt(DD,nearestNeighbour(DD).tour));if(sz<=EXACT_CAP)row.heldkarp=tm(()=>heldKarp(DD));return row;});setBench(rows);setBusy(false);},30);},[]);

  return(
    <div style={{display:"grid",gap:20}}>
      {tradeoff&&(
        <div style={{...card,padding:"15px 18px",background:R.red,display:"flex",gap:13,alignItems:"flex-start"}}>
          <AlertTriangle size={20} color="#fff" style={{flexShrink:0,marginTop:2}}/>
          <div style={{fontSize:13.5,lineHeight:1.6,color:"#fff",fontWeight:600}}>
            <span style={{fontWeight:900,textTransform:"uppercase",letterSpacing:.4}}>The shortest route is not the best route.</span>{" "}
            {shortest.name} rides the fewest kilometres ({km(shortest.length).toFixed(1)} km), but {fastestSla.name} restores the last critical site <b>{fmtMin((shortest.sla??0)-(fastestSla.sla??0))} sooner</b> ({fmtMin(fastestSla.sla)} vs {fmtMin(shortest.sla)}). When downtime costs SLA money, distance alone is the wrong objective — route <i>analysis</i> matters as much as route <i>design</i>.
          </div>
        </div>
      )}
      <div style={{display:"grid",gridTemplateColumns:cols2,gap:20}}>
        <div style={{...card,overflow:"hidden"}}>
          <div style={{padding:"12px 15px",borderBottom:BORDER,fontSize:14,fontWeight:900,textTransform:"uppercase",letterSpacing:.4,background:R.paper}}>Head-to-head · {n} stops</div>
          <table style={{width:"100%",borderCollapse:"collapse",fontSize:12.5}}>
            <thead><tr style={{color:R.mut,fontSize:10,textTransform:"uppercase",letterSpacing:.7,fontFamily:"var(--mono)"}}><th style={th}>Engine</th><th style={{...th,textAlign:"right"}}>km</th><th style={{...th,textAlign:"right"}}>Ratio</th>{hasSla&&<th style={{...th,textAlign:"right"}}>SLA</th>}<th style={{...th,textAlign:"right"}}>ms</th></tr></thead>
            <tbody>{[...compare].sort((a,b)=>a.length-b.length).map(c=>(
              <tr key={c.key} className="rrow" style={{borderTop:`1px solid #eceadf`}}>
                <td style={{...td,fontWeight:800,textTransform:"uppercase",letterSpacing:.2}}><span style={{display:"inline-block",width:9,height:9,borderRadius:2,background:c.color,border:`1.5px solid ${R.ink}`,marginRight:8}}/>{c.name}{c.length===best&&<Trophy size={12} color={R.red} style={{marginLeft:6,verticalAlign:"middle"}}/>}</td>
                <td style={{...td,textAlign:"right",fontFamily:"var(--mono)",fontWeight:700}}>{km(c.length).toFixed(1)}</td>
                <td style={{...td,textAlign:"right",fontFamily:"var(--mono)",fontWeight:700,color:c.ratio?(c.ratio<=1.0001?R.green:c.color):R.mut2}}>{c.ratio?`${c.ratio.toFixed(3)}×`:"—"}</td>
                {hasSla&&<td style={{...td,textAlign:"right",fontFamily:"var(--mono)",fontWeight:700,color:c.sla===fastestSla?.sla?R.green:R.ink}}>{c.sla!=null?fmtMin(c.sla):"—"}</td>}
                <td style={{...td,textAlign:"right",fontFamily:"var(--mono)",color:R.mut}}>{c.runtimeMs.toFixed(2)}</td>
              </tr>))}</tbody>
          </table>
          <div style={{padding:"11px 15px",fontSize:11.5,color:R.mut,borderTop:BORDER,lineHeight:1.55,fontWeight:600,background:R.paper}}>{optimal?<>Ratio measured against the true optimum from Held–Karp — every approximation sits under its proven bound.</>:<>With more than {EXACT_CAP} stops the optimum is unknowable in reasonable time — exactly where provable approximations earn their keep.</>}</div>
        </div>
        <div style={{...card,padding:"13px 8px 10px 0"}}>
          <div style={{padding:"0 15px 8px",fontSize:14,fontWeight:900,textTransform:"uppercase",letterSpacing:.4}}>Observed ratio vs proven bounds</div>
          {optimal?(
            <ResponsiveContainer width="100%" height={295}>
              <BarChart data={[...compare].filter(c=>c.ratio).sort((a,b)=>a.ratio-b.ratio)} margin={{top:10,right:18,left:4,bottom:28}}>
                <CartesianGrid stroke="#eceadf" vertical={false}/><XAxis dataKey="short" stroke={R.mut} tick={{fontSize:10.5,fontFamily:"var(--mono)",fontWeight:700}} angle={-15} dy={10} interval={0}/><YAxis stroke={R.mut} tick={{fontSize:10,fontFamily:"var(--mono)"}} domain={[1,"auto"]} width={36}/><Tooltip contentStyle={tip} formatter={v=>[`${v.toFixed(3)}×`,"ratio"]}/>
                <ReferenceLine y={1.5} stroke="#7048e8" strokeWidth={2} strokeDasharray="6 4" label={{value:"1.5 CHRISTOFIDES",fill:"#7048e8",fontSize:9.5,fontWeight:800,position:"insideTopRight"}}/>
                <ReferenceLine y={2.0} stroke={R.green} strokeWidth={2} strokeDasharray="6 4" label={{value:"2.0 MST",fill:R.green,fontSize:9.5,fontWeight:800,position:"insideTopRight"}}/>
                <ReferenceLine y={1.0} stroke={R.blue} strokeWidth={2} label={{value:"OPT",fill:R.blue,fontSize:9.5,fontWeight:800,position:"insideBottomRight"}}/>
                <Bar dataKey="ratio" radius={[4,4,0,0]} stroke={R.ink} strokeWidth={2}>{[...compare].filter(c=>c.ratio).sort((a,b)=>a.ratio-b.ratio).map(c=><Cell key={c.key} fill={c.color}/>)}</Bar>
              </BarChart>
            </ResponsiveContainer>
          ):<div style={{height:295,display:"grid",placeItems:"center",color:R.mut,fontSize:13,textAlign:"center",padding:20,fontWeight:600}}>Use ≤ {EXACT_CAP} stops to reveal exact ratios against the optimum.</div>}
        </div>
      </div>
      <div>
        <div style={{display:"flex",alignItems:"center",gap:14,marginBottom:16}}>
          <button onClick={runBench} disabled={busy} style={btnYellow}><Trophy size={15}/>{busy?"Running…":bench?"Re-run scaling experiment":"Run scaling experiment"}</button>
          <span style={{fontSize:12.5,color:R.mut,fontWeight:600}}>Times every engine on growing maps — the empirical face of the complexity classes.</span>
        </div>
        {bench?(
          <div style={{display:"grid",gridTemplateColumns:cols2,gap:20}}>
            <div style={{...card,padding:"14px 8px 8px 0"}}>
              <div style={{padding:"0 15px 8px",fontSize:13,fontWeight:900,textTransform:"uppercase",letterSpacing:.4}}>Runtime — all engines (ms)</div>
              <ResponsiveContainer width="100%" height={285}><LineChart data={bench} margin={{top:8,right:18,left:4,bottom:6}}><CartesianGrid stroke="#eceadf" vertical={false}/><XAxis dataKey="n" stroke={R.mut} tick={{fontSize:10,fontFamily:"var(--mono)"}}/><YAxis stroke={R.mut} tick={{fontSize:10,fontFamily:"var(--mono)"}} width={44}/><Tooltip contentStyle={tip}/><Legend wrapperStyle={{fontSize:11,fontFamily:"var(--mono)",fontWeight:700}}/>{["heldkarp","christofides","twoopt","mst","nn"].map(k=><Line key={k} type="monotone" dataKey={k} name={ALG[k].short} stroke={ALG[k].color} strokeWidth={2.4} dot={{r:2.5,strokeWidth:1.5,stroke:R.ink}} connectNulls/>)}</LineChart></ResponsiveContainer>
            </div>
            <div style={{...card,padding:"14px 8px 8px 0"}}>
              <div style={{padding:"0 15px 8px",fontSize:13,fontWeight:900,textTransform:"uppercase",letterSpacing:.4}}>Held–Karp — the exponential wall</div>
              <ResponsiveContainer width="100%" height={285}><LineChart data={bench.filter(r=>r.heldkarp!=null)} margin={{top:8,right:18,left:4,bottom:6}}><CartesianGrid stroke="#eceadf" vertical={false}/><XAxis dataKey="n" stroke={R.mut} tick={{fontSize:10,fontFamily:"var(--mono)"}}/><YAxis stroke={R.mut} tick={{fontSize:10,fontFamily:"var(--mono)"}} width={44}/><Tooltip contentStyle={tip}/><Line type="monotone" dataKey="heldkarp" name="O(2ⁿ·n²)" stroke={R.blue} strokeWidth={2.6} dot={{r:3,strokeWidth:1.5,stroke:R.ink}}/></LineChart></ResponsiveContainer>
              <div style={{padding:"4px 15px 8px",fontSize:11.5,color:R.mut,lineHeight:1.5,fontWeight:600}}>Each extra stop roughly doubles the work — the single chart that justifies this whole project.</div>
            </div>
          </div>
        ):<div style={{height:270,display:"grid",placeItems:"center",color:R.mut,border:`2px dashed ${R.ink}`,borderRadius:10,background:"#fff",fontWeight:700,textTransform:"uppercase",letterSpacing:.5,fontSize:12}}>Run the experiment to chart runtime vs map size</div>}
      </div>
    </div>
  );
}

/* ---------------- HOW IT WORKS ---------------- */
function Notes({onClose}){
  return(
    <div onClick={onClose} style={{position:"fixed",inset:0,background:"rgba(22,19,16,0.45)",zIndex:50,display:"flex",justifyContent:"flex-end"}}>
      <div onClick={e=>e.stopPropagation()} style={{width:"min(560px,94vw)",height:"100%",overflowY:"auto",background:R.bg,borderLeft:BORDER,padding:"26px 26px"}}>
        <div style={{display:"flex",justifyContent:"space-between",alignItems:"center",marginBottom:16}}>
          <h2 style={{fontSize:22,fontWeight:900,margin:0,textTransform:"uppercase",letterSpacing:.3}}>How Odyssey works</h2>
          <button onClick={onClose} style={{...btn,width:36,padding:0}}><X size={15}/></button>
        </div>
        <div style={{...card,background:R.yellow,padding:"15px 17px",marginBottom:18}}>
          <div style={{fontSize:14,fontWeight:900,textTransform:"uppercase",letterSpacing:.4,marginBottom:6}}>The real problem</div>
          <p style={{fontSize:13,lineHeight:1.62,margin:0,fontWeight:600}}>Every day, telecom field teams, delivery riders, and ambulance restocking crews face the same question: <b>in what order do I visit all my sites?</b> This is the Travelling Salesman Problem, and it is <b>NP-hard</b> — no known algorithm solves it efficiently as the site count grows. Odyssey implements one solver from every major algorithm-design paradigm, runs them on real Bengaluru scenarios, and measures them: against the proven optimum, against their theoretical bounds, and against operational metrics like fuel cost and critical-site restore time. Design <i>and</i> analysis.</p>
        </div>
        <div style={{...card,background:R.red,padding:"15px 17px",marginBottom:18}}>
          <div style={{fontSize:14,fontWeight:900,textTransform:"uppercase",letterSpacing:.4,marginBottom:6,color:"#fff"}}>The twist nobody teaches</div>
          <p style={{fontSize:13,lineHeight:1.62,margin:0,fontWeight:600,color:"#fff"}}>Textbooks stop at "minimize distance." Odyssey adds a second objective — time until critical sites are restored — and shows that the shortest route often makes a DOWN tower wait <i>longer</i>. Two valid objectives, two different winners. That trade-off is the difference between an algorithms assignment and an engineering decision.</p>
        </div>
        {ORDER.map(k=>{const a=ALG[k],Icon=a.icon;return(
          <div key={k} style={{...cardFlat,padding:"14px 16px",marginBottom:12,borderLeft:`6px solid ${a.color}`}}>
            <div style={{display:"flex",alignItems:"center",gap:9,marginBottom:7}}><Icon size={16} color={a.color}/><span style={{fontSize:14.5,fontWeight:900,textTransform:"uppercase",letterSpacing:.2}}>{a.name}</span></div>
            <div style={{display:"flex",gap:6,flexWrap:"wrap",marginBottom:9}}>
              {[a.paradigm,a.complexity,a.guarantee].map((t,i)=>(<span key={i} style={{fontSize:10,fontFamily:"var(--mono)",fontWeight:700,color:i===1?R.ink:"#fff",background:i===1?R.yellow:a.color,padding:"3px 8px",borderRadius:5,border:`1.5px solid ${R.ink}`}}>{t}</span>))}
            </div>
            <p style={{color:R.mut,fontSize:12.4,lineHeight:1.56,margin:0,fontWeight:500}}>{a.long}</p>
          </div>);})}
      </div>
    </div>
  );
}
