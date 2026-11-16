'use strict';
const $ = id => document.getElementById(id);
const palette = ['#2463c7','#d07a25','#179475','#a05bb8','#ce526d','#4d8b9a','#848b35','#7160a0'];
const labels = {'actor/distillation/abs_loss':'蒸馏 loss','actor/loss':'训练 loss','actor/grad_norm':'梯度范数','response_length/mean':'平均回答长度','timing_s/step':'每步耗时（秒）'};
let index, selected = new Set(), cache = new Map(), renderToken = 0;
const charts = [];
const fmt = n => !Number.isFinite(n) ? '—' : Math.abs(n)>=10000 ? n.toLocaleString('zh-CN',{maximumFractionDigits:0}) : Math.abs(n)>=10 ? n.toFixed(2) : Number(n.toPrecision(5)).toString();
function shortName(name){const p=name.split('/');return p.slice(0,1).concat(p.slice(-2)).join(' / ');}
function el(tag,text,className){const node=document.createElement(tag);if(text!==undefined)node.textContent=text;if(className)node.className=className;return node;}
async function dataset(run){if(!cache.has(run.id))cache.set(run.id,fetch('data/'+run.id+'.json').then(r=>{if(!r.ok)throw Error('数据读取失败');return r.json();}).catch(e=>{cache.delete(run.id);throw e;}));return cache.get(run.id);}
function listRuns(){
  const query=$('search').value.toLowerCase(), machine=$('machine').value, kind=$('kind').value;
  const filtered=index.runs.filter(r=>(!machine||r.machine===machine)&&(!kind||r.kind===kind)&&r.name.toLowerCase().includes(query));
  $('runs').replaceChildren();
  for(const group of [...new Set(filtered.map(r=>r.machine))]){
    const section=el('div',undefined,'run-group');section.append(el('div',group,'group-title'));
    for(const run of filtered.filter(r=>r.machine===group)){
      const label=el('label',undefined,'run-option'), input=el('input');input.type='checkbox';input.checked=selected.has(run.id);
      const text=el('span',run.name.split('/').slice(1).join(' / '));text.append(el('small',`${run.kind} · 最后步数 ${run.lastStep} · ${run.tags.length} 项指标`));
      input.addEventListener('change',()=>{input.checked?selected.add(run.id):selected.delete(run.id);render();});label.append(input,text);section.append(label);
    }$('runs').append(section);
  }
  $('counts').textContent=`显示 ${filtered.length} / ${index.runs.length} 组记录`;
}
function smooth(points,weight){let last;return points.map(([x,y])=>{last=last===undefined?y:weight*last+(1-weight)*y;return[x,last];});}
function plot(canvasId,detailId,series,tag){
  const canvas=$(canvasId), ctx=canvas.getContext('2d'), detail=$(detailId), W=canvas.clientWidth, H=canvas.clientHeight, dpr=window.devicePixelRatio||1;
  canvas.width=Math.round(W*dpr);canvas.height=Math.round(H*dpr);ctx.scale(dpr,dpr);
  const pad={l:63,r:16,t:12,b:34}, points=series.flatMap(s=>s.points);
  ctx.font='11px system-ui';ctx.fillStyle='#768699';
  if(!points.length){ctx.textAlign='center';ctx.fillText(selected.size?'所选实验没有这项指标':'从左侧选择实验',W/2,H/2);detail.textContent='';canvas.onpointermove=null;return;}
  let minX=Math.min(...points.map(p=>p[0])),maxX=Math.max(...points.map(p=>p[0]));
  let minY=Math.min(...points.map(p=>p[1])),maxY=Math.max(...points.map(p=>p[1]));
  if(minX===maxX){minX=Math.max(0,minX-1);maxX+=1;}const yp=(maxY-minY)*.08||Math.abs(maxY)*.1||1;minY-=yp;maxY+=yp;
  const x=v=>pad.l+(v-minX)/(maxX-minX)*(W-pad.l-pad.r),y=v=>H-pad.b-(v-minY)/(maxY-minY)*(H-pad.t-pad.b);
  for(let i=0;i<=4;i++){
    const value=minY+(maxY-minY)*i/4,py=y(value);ctx.strokeStyle='#e9edf3';ctx.lineWidth=1;ctx.beginPath();ctx.moveTo(pad.l,py);ctx.lineTo(W-pad.r,py);ctx.stroke();ctx.fillStyle='#768699';ctx.textAlign='right';ctx.fillText(fmt(value),pad.l-8,py+4);
    const xv=minX+(maxX-minX)*i/4;ctx.textAlign='center';ctx.fillText(Math.round(xv).toString(),x(xv),H-13);
  }
  ctx.fillStyle='#768699';ctx.textAlign='right';ctx.fillText('STEP',W-pad.r,H-1);
  for(const s of series){ctx.strokeStyle=s.color;ctx.fillStyle=s.color;ctx.lineWidth=1.8;ctx.beginPath();let prev;
    s.points.forEach(([a,b],i)=>{const gap=prev!==undefined&&a-prev>1&&!tag.startsWith('eval/');if(i===0||gap)ctx.moveTo(x(a),y(b));else ctx.lineTo(x(a),y(b));prev=a;});ctx.stroke();
    if(s.points.length<=12){for(const[a,b]of s.points){ctx.beginPath();ctx.arc(x(a),y(b),3.4,0,Math.PI*2);ctx.fill();}}
  }
  canvas.onpointermove=e=>{const px=e.clientX-canvas.getBoundingClientRect().left,step=minX+(px-pad.l)/(W-pad.l-pad.r)*(maxX-minX);detail.replaceChildren();for(const s of series){if(!s.points.length)continue;const p=s.points.reduce((a,b)=>Math.abs(a[0]-step)<Math.abs(b[0]-step)?a:b);const line=el('span',`${shortName(s.name)} · step ${p[0]}：${fmt(p[1])}`);line.style.marginRight='14px';detail.append(line);}};
  detail.textContent='移动到曲线上查看数值';
  charts.push({canvasId,detailId,series,tag});
}
async function render(){
  const token=++renderToken, runs=index.runs.filter(r=>selected.has(r.id));
  $('message').textContent=runs.length?'正在读取所选曲线…':'';
  try{
    const data=await Promise.all(runs.map(dataset));if(token!==renderToken)return;
    const tags=[...new Set(runs.flatMap(r=>r.tags))].sort();$('metrics').replaceChildren(...tags.map(t=>{const o=el('option');o.value=t;return o;}));
    const tag=$('metric').value.trim(), weight=Number($('smooth').value)/100;
    $('main-title').textContent=labels[tag]||tag||'选择主图指标';$('smooth-value').textContent=Math.round(weight*100)+'%';
    $('legend').replaceChildren(...runs.map((r,i)=>{const item=el('span',undefined,'legend-item'),swatch=el('i',undefined,'swatch');swatch.style.background=palette[i%palette.length];item.append(swatch,el('span',shortName(r.name)));return item;}));
    function series(key){return runs.map((r,i)=>{const raw=data[i].metrics[key]||[];return{name:r.name,color:palette[i%palette.length],points:smooth(raw,key.startsWith('eval/')?0:weight)};});}
    charts.length=0;plot('main-chart','main-detail',series(tag),tag);plot('grad-chart','grad-detail',series('actor/grad_norm'),'actor/grad_norm');plot('length-chart','length-detail',series('response_length/mean'),'response_length/mean');
    $('summary').replaceChildren(...runs.map((r,i)=>{const row=el('tr'),points=data[i].metrics[tag]||[];[r.name,r.lastStep,points.length?fmt(points.at(-1)[1]):'无此指标',r.tags.length].forEach(t=>row.append(el('td',t)));return row;}));
    const missing=runs.filter((_,i)=>!data[i].metrics[tag]);$('message').textContent=missing.length?`${missing.length} 组记录没有当前主指标；可在指标框中切换，评测分数以 eval/ 开头。`:'';
  }catch(e){$('message').textContent=e.message+'，请刷新后重试。';}
}
async function download(){
  const runs=index.runs.filter(r=>selected.has(r.id));if(!runs.length){$('message').textContent='请先选择实验。';return;}
  const data=await Promise.all(runs.map(dataset)),result={updated:index.updated,runs:runs.map((r,i)=>({name:r.name,...data[i]}))};
  const url=URL.createObjectURL(new Blob([JSON.stringify(result,null,2)],{type:'application/json'})),a=el('a');a.href=url;a.download='bridge-mopd-metrics.json';a.click();setTimeout(()=>URL.revokeObjectURL(url),1000);
}
async function init(){
  const response=await fetch('data/index.json');if(!response.ok)throw Error('实验索引读取失败');index=await response.json();
  $('snapshot').textContent=`${index.trainingCount} 组训练曲线 · ${index.runs.length} 组实验与分析记录 · ${index.updated}`;
  $('footer-count').textContent='H800 / B200 / 旧机器';
  for(const machine of [...new Set(index.runs.map(r=>r.machine))]){const option=el('option',machine);option.value=machine;$('machine').append(option);}
  const preferred=['H800/2026-11-14/1.7B_普通MOPD','H800/2026-11-14/1.7B_自适应渐进式教师'];
  for(const name of preferred){const run=index.runs.find(r=>r.name===name);if(run)selected.add(run.id);}
  if(!selected.size&&index.runs.length)selected.add(index.runs[0].id);
  for(const id of ['search','machine','kind'])$(id).addEventListener('input',listRuns);
  $('clear').addEventListener('click',()=>{selected.clear();listRuns();render();});$('metric').addEventListener('change',render);$('smooth').addEventListener('input',render);$('download').addEventListener('click',()=>download().catch(e=>$('message').textContent=e.message));
  let timer;window.addEventListener('resize',()=>{clearTimeout(timer);timer=setTimeout(render,150);});listRuns();await render();
}
init().catch(e=>{$('snapshot').textContent=e.message;});
