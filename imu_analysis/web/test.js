const grid = document.querySelector('#device-grid');
const template = document.querySelector('#device-template');
const note = document.querySelector('#live-note');
const summary = document.querySelector('#connection-summary');
const receiverControl = document.querySelector('#receiver-control');
const postureCalibrate = document.querySelector('#posture-calibrate');
let mode = 'motion';
let receiverRunning = false;
const formatSigned = (value, suffix) => `${value >= 0 ? '+' : ''}${Number(value).toFixed(1)}${suffix}`;
const formatDuration = (seconds) => {
  const value = Math.max(0, Math.round(Number(seconds) || 0));
  return `${String(Math.floor(value / 60)).padStart(2, '0')}:${String(value % 60).padStart(2, '0')}`;
};
function postureView(stream) {
  const posture = stream.posture || {};
  if (posture.state === 'calibrating') {
    return { label: '正在校准', confidence: `${Math.round((posture.calibration_progress || 0) * 100)}%`, duration: '保持自然坐姿' };
  }
  if (posture.state === 'calibration_failed') {
    return { label: '校准失败', confidence: '头部移动过多', duration: '请重新校准' };
  }
  if (posture.state === 'calibrated') {
    return { label: '校准完成', confidence: '等待分析窗口', duration: '00:00' };
  }
  if (posture.state === 'monitoring') {
    return {
      label: posture.posture_name_zh || posture.posture || '正常坐姿',
      confidence: `${Math.round((posture.confidence || 0) * 100)}%`,
      duration: formatDuration(posture.continuous_seconds),
    };
  }
  return { label: '等待坐姿模型', confidence: '—', duration: '00:00' };
}
function metricsFor(stream) {
  if (mode === 'motion') return [['Gyro X', formatSigned(stream.gyro_x_dps || 0, ' °/s')], ['Gyro Y', formatSigned(stream.gyro_y_dps || 0, ' °/s')], ['Gyro Z', formatSigned(stream.gyro_z_dps || 0, ' °/s')]];
  const posture = postureView(stream);
  return [['当前姿态', posture.label], ['模型置信度', posture.confidence], ['异常持续', posture.duration]];
}
function placeholder(index) { return { device: `设备 ${index + 1}`, state: 'waiting', gyro_x_dps: 0, gyro_y_dps: 0, gyro_z_dps: 0, pitch_degrees: 0, roll_degrees: 0, yaw_degrees: 0, sample_rate_hz: 0, last_sample_age_s: null, posture: {} }; }
function render(streams) {
  const twoStreams = [...streams.slice(0, 2)]; while (twoStreams.length < 2) twoStreams.push(placeholder(twoStreams.length));
  grid.replaceChildren(...twoStreams.map((stream, index) => {
    const node = template.content.firstElementChild.cloneNode(true); const isLive = stream.state === 'live'; node.classList.toggle('is-waiting', !isLive);
    node.querySelector('.device-name strong').textContent = `${stream.device}${index === 0 ? ' · 设备 A' : ' · 设备 B'}`; node.querySelector('.live-badge').textContent = isLive ? '实时' : '等待'; node.querySelector('.status-dot').classList.toggle('offline', !isLive);
    node.querySelector('.metrics').replaceChildren(...metricsFor(stream).map(([label, value]) => { const item = document.createElement('div'); item.innerHTML = `<dt>${label}</dt><dd>${value}</dd>`; return item; }));
    const visual = node.querySelector('.gyro-visual'); visual.style.setProperty('--pitch', `${Math.max(-45, Math.min(45, stream.pitch_degrees || 0))}deg`); visual.style.setProperty('--roll', `${Math.max(-45, Math.min(45, stream.roll_degrees || 0))}deg`); visual.style.setProperty('--yaw', `${Math.max(-45, Math.min(45, stream.yaw_degrees || 0))}deg`);
    node.querySelector('.stream-meta').textContent = stream.sample_rate_hz ? `${stream.sample_rate_hz} Hz · 数据流正常` : '等待实时数据'; node.querySelector('.updated-at').textContent = stream.last_sample_age_s === null ? '尚未收到样本' : `${stream.last_sample_age_s.toFixed(1)} 秒前更新`; return node;
  }));
}
async function refresh() { try { const response = await fetch('/api/live', { cache: 'no-store' }); if (!response.ok) throw new Error('live endpoint unavailable'); const live = await response.json(); const connected = live.streams.filter((stream) => stream.state === 'live').length; summary.textContent = `已连接 ${connected} / 2 个设备`; const first = live.streams[0]; const posture = first ? postureView(first) : null; note.textContent = connected ? (mode === 'motion' ? '同步采样中 · 每秒刷新 · 显示陀螺仪三轴角速度' : `坐姿模型实时测试 · ${posture.label} · 异常持续 ${posture.duration}`) : '等待实时蓝牙 IMU 数据…'; render(live.streams || []); } catch (_) { summary.textContent = '实时连接失败'; note.textContent = '无法读取 /api/live，确认实时接收程序和网页服务已启动。'; render([]); } }
async function refreshControl() { try { const response = await fetch('/api/control', { cache: 'no-store' }); const state = await response.json(); receiverRunning = Boolean(state.running); receiverControl.textContent = receiverRunning ? '停止蓝牙采集' : '启动蓝牙采集'; receiverControl.classList.toggle('is-running', receiverRunning); } catch (_) { receiverControl.textContent = '控制服务不可用'; receiverControl.disabled = true; } }
receiverControl.addEventListener('click', async () => { receiverControl.disabled = true; receiverControl.textContent = receiverRunning ? '正在停止…' : '正在启动…'; try { const response = await fetch('/api/control', { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify({ action: receiverRunning ? 'stop' : 'start' }) }); if (!response.ok) throw new Error('control failed'); await refreshControl(); await refresh(); } catch (_) { receiverControl.textContent = '操作失败，请重试'; } finally { receiverControl.disabled = false; } });
postureCalibrate.addEventListener('click', async () => {
  postureCalibrate.disabled = true;
  postureCalibrate.textContent = '正在下发校准…';
  try {
    const response = await fetch('/api/posture/calibrate', { method: 'POST' });
    if (!response.ok) throw new Error('calibration failed');
    postureCalibrate.textContent = '校准中，请保持自然坐姿';
    await refresh();
  } catch (_) {
    postureCalibrate.textContent = '校准失败，请重试';
  } finally {
    window.setTimeout(() => {
      postureCalibrate.disabled = false;
      postureCalibrate.textContent = '10 秒重新校准';
    }, 1200);
  }
});
document.querySelectorAll('[data-mode]').forEach((button) => button.addEventListener('click', () => { mode = button.dataset.mode; document.querySelectorAll('[data-mode]').forEach((item) => item.classList.toggle('active', item === button)); postureCalibrate.classList.toggle('is-hidden', mode !== 'posture'); refresh(); }));
refresh(); refreshControl(); setInterval(refresh, 1000); setInterval(refreshControl, 2000);
