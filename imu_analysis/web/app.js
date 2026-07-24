const input = document.querySelector('#file-input');
const duration = document.querySelector('#total-duration');
const summary = document.querySelector('#today-summary');
const statusTitle = document.querySelector('#status-title');
const statusDetail = document.querySelector('#status-detail');
const list = document.querySelector('#training-list');
const postureState = document.querySelector('#posture-state');
const postureAngle = document.querySelector('#posture-angle');
const postureDetail = document.querySelector('#posture-detail');
const pet = document.querySelector('#posture-pet');

async function refreshLiveData() {
  try {
    const response = await fetch('/api/live', { cache: 'no-store' });
    const live = await response.json();
    const stream = live.streams?.[0];
    if (!stream) return;
    statusTitle.textContent = `${stream.device} 实时连接中`;
    const warning = stream.experimental ? ` · ${stream.warning || '实验模型'}` : '';
    statusDetail.textContent = `${stream.label} · 运动置信度 ${stream.exercise_probability}% · ${stream.workout_state || '等待'}${warning}`;
    postureState.textContent = stream.workout_state === 'active' ? '运动中' : '当前姿态监测中';
    postureAngle.textContent = `信号质量 ${stream.signal_quality || '等待'} · 来自 ${stream.device}`;
    postureDetail.textContent = `实时数据已更新 · 最近样本 ${stream.last_sample_age_s} 秒前${stream.posture?.experimental ? ` · ${stream.posture.warning || '实验姿态模型'}` : ''}`;
  } catch (_) { /* Offline upload continues to work when no receiver is running. */ }
}

function formatDuration(seconds) {
  const rounded = Math.max(0, Math.round(seconds || 0));
  const minutes = Math.floor(rounded / 60);
  const remaining = rounded % 60;
  if (minutes && remaining) return `${minutes} 分 ${remaining} 秒`;
  if (minutes) return `${minutes} 分钟`;
  return `${remaining} 秒`;
}

function activityText(activity) {
  if (activity.kind !== 'strength') return `${activity.action} ${formatDuration(activity.duration_seconds)}`;
  return `${activity.action} ${activity.sets} 组`;
}

function renderSessions(sessions) {
  if (!sessions.length) {
    list.innerHTML = '<p class="empty-state">本次未检测到训练记录</p>';
    return;
  }
  list.innerHTML = sessions.slice().reverse().map((session, index) => {
    const activities = session.activities?.map(activityText).join(' · ') || `置信度 ${session.confidence || 0}%`;
    const clock = session.start && session.end ? `${session.start}–${session.end}` : `本次训练 ${sessions.length - index}`;
    return `
    <article class="training-item">
      <div><strong>${clock}</strong><span>${formatDuration(session.duration_seconds)} · ${activities}</span></div>
      <span class="chevron">›</span>
    </article>`;
  }).join('');
}

async function refreshDailySummary() {
  try {
    const response = await fetch('/api/daily', { cache: 'no-store' });
    if (!response.ok) return;
    const today = await response.json();
    if (!today.sessions?.length) return;
    duration.textContent = formatDuration(today.total_workout_seconds);
    summary.textContent = `${today.session_count} 段训练 · 有效运动 ${formatDuration(today.active_seconds)}`;
    renderSessions(today.sessions);
  } catch (_) { /* A missing local database is equivalent to no workouts yet. */ }
}

function setClock() {
  const time = new Intl.DateTimeFormat('zh-CN', { hour: '2-digit', minute: '2-digit', hour12: false }).format(new Date());
  document.querySelectorAll('#clock, .clock').forEach((clock) => { clock.textContent = time; });
}

input.addEventListener('change', async () => {
  const file = input.files[0];
  if (!file) return;
  duration.textContent = '正在分析';
  summary.textContent = file.name;
  statusTitle.textContent = '模型推理中';
  statusDetail.textContent = '正在提取 IMU 特征并识别训练片段';
  try {
    const content = await file.text();
    const response = await fetch('/api/analyze', {
      method: 'POST', headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ filename: file.name, content }),
    });
    const result = await response.json();
    if (!response.ok) throw new Error(result.message || '推理失败');
    duration.textContent = formatDuration(result.exercise_duration_seconds);
    summary.textContent = `${result.session_count} 段训练 · ${result.windows} 个窗口已自动整理`;
    statusTitle.textContent = result.status;
    statusDetail.textContent = `平均运动置信度 ${result.exercise_probability}% · ${result.filename}${result.experimental ? ` · ${result.warning || '实验模型'}` : ''}`;
    renderSessions(result.sessions);
    postureState.textContent = result.posture.state;
    postureAngle.textContent = result.posture.angle_degrees === null ? '未读取到姿态角' : `姿态角 ${result.posture.angle_degrees}° · 运动置信度 ${result.exercise_probability}%`;
    postureDetail.textContent = result.posture.detail;
    pet.dataset.mood = result.posture.mood;
  } catch (error) {
    duration.textContent = '分析失败';
    summary.textContent = '请确认文件是耳机导出的 CSV 或 TXT 数据';
    statusTitle.textContent = '无法完成推理';
    statusDetail.textContent = error.message;
    renderSessions([]);
  } finally { input.value = ''; }
});

document.querySelectorAll('[data-target]').forEach((button) => {
  button.addEventListener('click', () => {
    document.querySelectorAll('[data-view]').forEach((view) => view.classList.toggle('active', view.dataset.view === button.dataset.target));
    document.querySelectorAll('[data-target]').forEach((item) => item.classList.toggle('active', item === button));
  });
});
document.querySelector('#calibrate').addEventListener('click', () => { postureDetail.textContent = '已记录当前姿态作为本次浏览器会话的参考。'; });

setClock();
refreshLiveData();
refreshDailySummary();
setInterval(refreshLiveData, 1000);
setInterval(refreshDailySummary, 5000);
