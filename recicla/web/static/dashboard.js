
const $ = id => document.getElementById(id);
const settingsDialog = $('settings');
const hist = $('history');
const video = $('video');
let data = {state: 'idle'};
let polling = false;
let fetchingFrame = false;
let previousFrame;
let frameGeneration = 0;
let displayedFrameMode;
let overrideRequest = Promise.resolve();
const overrideKeys = new Map([['0', null], ['1', 'garrafa'], ['2', 'lata']]);



async function api(url, options) {
    const response = await fetch(url, options);
    if (!response.ok) {
        const body = await response.json().catch(() => ({}));
        throw new Error(body.error || 'Falha de conexão com o serviço local.');
    }
    return response;
}

async function act(action, extra = {}) {
    setMenu(false);
    try {
        await api('/api/action', {
            method: 'POST',
            headers: {'Content-Type': 'application/json'},
            body: JSON.stringify({action, ...extra})
        });
        return true;
    } catch (error) {
        notify(error.message);
        return false;
    }
}

async function exportSession() {
    setMenu(false);
    try {
        const response = await api('/api/export');
        const name = response.headers.get('Content-Disposition')?.match(/filename="?([^";]+)/)?.[1];
        const url = URL.createObjectURL(await response.blob());
        const link = document.createElement('a');
        link.href = url;
        link.download = name || 'sessao_resumo.csv';
        document.body.append(link);
        link.click();
        link.remove();
        setTimeout(() => URL.revokeObjectURL(url), 1000);
    } catch (error) {
        notify(error.message);
    }
}

function openSettings() {
    if (!data.settings) return;
    if (data.state !== 'idle') {
        notify('Encerre a câmera primeiro.');
        return;
    }
    $('source').value = data.settings.camera.source;
    $('path').value = data.settings.model.path;
    $('conf').value = data.settings.model.conf;
    settingsDialog.showModal();
    setMenu(false);
}

function openHistory() {
    hist.showModal();
    setMenu(false);
}

$('settingsForm').onsubmit = async event => {
    event.preventDefault();
    const saved = await act('settings', {
        source: $('source').value,
        path: $('path').value,
        conf: $('conf').value
    });
    if (saved) settingsDialog.close();
};

function text(id, value) {
    $(id).textContent = value;
}

function clearFrame() {
    frameGeneration++;
    video.hidden = true;
    video.removeAttribute('src');
    $('placeholder').hidden = false;
    if (previousFrame) URL.revokeObjectURL(previousFrame);
    previousFrame = undefined;
    displayedFrameMode = undefined;
}

function updateHud() {
    const model = data.model?.is_demo ? 'MODELO DEMO · COCO'
        : data.model ? data.model.device.toUpperCase() + ' · YOLO' : 'Sistema pronto';
    text('hud', model);
}

function setClassOverride(material) {
    overrideRequest = overrideRequest.then(async () => {
        await api('/api/action', {
            method: 'POST',
            headers: {'Content-Type': 'application/json'},
            body: JSON.stringify({action: 'class_override', material})
        });
        await poll();
    }).catch(error => notify(error.message));
}

async function poll() {
    if (polling) return;
    polling = true;
    try {
        data = await (await api('/api/state', {cache: 'no-store'})).json();
        const stats = data.stats;
        updateStartup();
        const sameSession = countedSession === stats.started_at && stats.total >= lastTotal;
        animateCount($('total'), stats.total, sameSession);
        text('weight', stats.total_weight_kg < 1
            ? Math.round(stats.total_weight_kg * 1000) + ' g'
            : stats.total_weight_kg.toFixed(2) + ' kg');
        text('value', stats.total_value.toLocaleString('pt-BR', {style: 'currency', currency: 'BRL'}));
        text('state', {idle: 'Em espera', loading: 'Carregando', running: 'Ao vivo', paused: 'Pausado'}[data.state]);
        text('status', data.status);
        if (data.error && data.error !== lastError && startupFinished) notify(data.error);
        lastError = data.error;
        text('perf', data.fps.toFixed(1) + ' FPS · ' + Math.round(data.ms) + ' ms');
        updateHud();
        $('start').disabled = data.state !== 'idle';
        $('stop').disabled = data.state === 'idle';
        $('pause').disabled = !['running', 'paused'].includes(data.state);
        text('pause', data.state === 'paused' ? 'Retomar' : 'Pausar');
        updateMaterials(data.materials, stats, sameSession);
        countedSession = stats.started_at;
        lastTotal = stats.total;
        $('events').replaceChildren(...data.events.slice().reverse().map(event => {
            const row = document.createElement('tr');
            const material = data.materials.find(item => item.id === event.material);
            for (const value of [new Date(event.timestamp * 1000).toLocaleTimeString(),
                material?.label || event.material, event.conf === null ? '—' : Math.round(event.conf * 100) + '%']) {
                const cell = document.createElement('td');
                cell.textContent = value;
                row.append(cell);
            }
            return row;
        }));
        if (['idle', 'loading'].includes(data.state)) clearFrame();
    } catch (error) {
        text('status', 'Conexão local indisponível');
        if (!startupFinished && animationFinished && performance.now() - pageOpened > 5000) {
            showStartupError('Conexão local indisponível. Verifique se a EcoVision está em execução.');
        }
    } finally {
        polling = false;
    }
}

async function frames() {
    const mode = data.class_override;
    const pausedModeChanged = data.state === 'paused' && displayedFrameMode !== mode;
    if (fetchingFrame || (data.state !== 'running' && !pausedModeChanged)) return;
    fetchingFrame = true;
    const generation = frameGeneration;
    try {
        const response = await api('/api/frame', {cache: 'no-store'});
        if (response.status === 204) return;
        const image = await response.blob();
        if (generation !== frameGeneration || !['running', 'paused'].includes(data.state)) return;
        const url = URL.createObjectURL(image);
        video.src = url;
        video.hidden = false;
        $('placeholder').hidden = true;
        if (previousFrame) URL.revokeObjectURL(previousFrame);
        previousFrame = url;
        displayedFrameMode = mode;
    } catch (error) {
        text('status', error.message);
    } finally {
        fetchingFrame = false;
    }
}

const pageOpened = performance.now();
const reducedMotion = matchMedia('(prefers-reduced-motion: reduce)');
let introRequested = false;
let animationFinished = false;
let startupFinished = false;
let countedSession;
let lastTotal = 0;
let lastError;
let notificationTimer;
let cursorTimer;
let presentationPending = false;
const counts = new WeakMap();
const materialRows = new Map();

function finishSplash() {
    if (startupFinished) return;
    startupFinished = true;
    $('splash').classList.add('leaving');
    $('splash').inert = true;
    $('dashboard').inert = false;
    setTimeout(() => { $('splash').hidden = true; }, reducedMotion.matches ? 0 : 480);
}

function showStartupError(message) {
    text('startupMessage', message);
    $('startupMessage').hidden = false;
    $('startupContinue').hidden = false;
}

function updateStartup() {
    if (startupFinished || !animationFinished) return;
    if (data.startup === 'error') {
        showStartupError(data.startup_error);
    } else {
        $('startupMessage').hidden = true;
        $('startupContinue').hidden = true;
        if (data.startup === 'ready') finishSplash();
    }
}

function startIntro() {
    if (introRequested || startupFinished) return;
    introRequested = true;
    $('splashPrompt').hidden = true;
    setTimeout(() => {
        $('splash').classList.remove('waiting');
        if (reducedMotion.matches) {
            animationFinished = true;
            updateStartup();
        }
    }, 600);
}

function setupScanner() {
    const svg = $('splashMark').querySelector('svg');
    const eye = svg?.querySelector('.eye, #ecovision-eye');
    if (!eye || !svg.viewBox.baseVal.width) return;
    const ns = 'http://www.w3.org/2000/svg';
    const {x, y, width, height} = svg.viewBox.baseVal;
    const clip = document.createElementNS(ns, 'clipPath');
    clip.id = 'ecovision-scanner-clip';
    const outline = eye.cloneNode(false);
    outline.removeAttribute('id');
    outline.removeAttribute('class');
    outline.style.fill = 'white';
    clip.append(outline);
    const interior = document.createElementNS(ns, 'g');
    interior.setAttribute('clip-path', 'url(#ecovision-scanner-clip)');
    const beam = document.createElementNS(ns, 'rect');
    beam.classList.add('scanner-beam');
    beam.style.setProperty('--scan-distance', height + 'px');
    beam.setAttribute('x', x);
    beam.setAttribute('y', y);
    beam.setAttribute('width', width);
    beam.setAttribute('height', Math.max(1, height * .025));
    eye.setAttribute('pathLength', '1');
    interior.append(beam);
    svg.append(clip, interior);
}

function setMenu(open) {
    if (!open && $('menu').contains(document.activeElement)) $('menuBtn').focus();
    $('menuBtn').setAttribute('aria-expanded', String(open));
    $('menu').inert = !open;
    $('menu').classList.toggle('open', open);
}

function notify(message) {
    clearTimeout(notificationTimer);
    text('notification', message);
    $('notification').classList.add('visible');
    notificationTimer = setTimeout(() => $('notification').classList.remove('visible'), 4500);
}

function animateCount(element, value, animate) {
    const previous = counts.get(element);
    if (previous?.target === value && animate) return;
    if (previous) cancelAnimationFrame(previous.frame);
    const count = {target: value, displayed: value};
    counts.set(element, count);
    if (!animate || reducedMotion.matches || !previous || value <= previous.target) {
        element.textContent = value;
        return;
    }
    const startValue = previous.displayed;
    const start = performance.now();
    const card = element.closest('.stat, .material-count');
    card.getAnimations().filter(animation => animation.id === 'count-pulse')
        .forEach(animation => animation.cancel());
    card.animate([{boxShadow: 'inset 0 0 0 2px #91e8c3a0'}, {boxShadow: 'inset 0 0 0 2px #91e8c300'}],
        {id: 'count-pulse', duration: 620, easing: 'ease-out'});
    function tick(now) {
        const progress = Math.min(1, (now - start) / 320);
        count.displayed = Math.round(startValue + (value - startValue) * (1 - (1 - progress) ** 3));
        element.textContent = count.displayed;
        if (progress < 1) count.frame = requestAnimationFrame(tick);
    }
    count.displayed = startValue;
    count.frame = requestAnimationFrame(tick);
}

function updateMaterials(materials, stats, animate) {
    for (const material of materials) {
        if (!materialRows.has(material.id)) {
            const row = document.createElement('div');
            row.className = 'row material-count';
            const label = document.createElement('span');
            const count = document.createElement('strong');
            label.textContent = material.label;
            row.append(label, count);
            $('materials').append(row);
            materialRows.set(material.id, count);
        }
        animateCount(materialRows.get(material.id), stats.by_material[material.id].count, animate);
    }
}

function revealCursor() {
    clearTimeout(cursorTimer);
    document.body.classList.remove('cursor-idle');
    if (document.body.classList.contains('presentation')) {
        cursorTimer = setTimeout(() => document.body.classList.add('cursor-idle'), 3000);
    }
}

async function enterPresentation() {
    setMenu(false);
    if (presentationPending || document.fullscreenElement) return;
    presentationPending = true;
    try {
        await document.documentElement.requestFullscreen();
    } catch {
        notify('O navegador não conseguiu abrir a tela cheia.');
    } finally {
        presentationPending = false;
    }
}

document.addEventListener('fullscreenchange', () => {
    document.body.classList.toggle('presentation', document.fullscreenElement === document.documentElement);
    revealCursor();
});
document.addEventListener('pointermove', revealCursor, {passive: true});
document.addEventListener('keydown', event => {
    revealCursor();
    if (event.defaultPrevented) return;
    if (event.key === 'Escape') {
        if (document.fullscreenElement === document.documentElement) {
            document.exitFullscreen().catch(() => {});
        } else {
            setMenu(false);
        }
    }
    if (!introRequested && (event.key === 'Enter' || event.key === ' ')) {
        event.preventDefault();
        startIntro();
        return;
    }
    if (event.repeat || event.isComposing || event.altKey || event.metaKey || event.ctrlKey ||
        settingsDialog.open || hist.open || event.target.closest('input, textarea, select, [contenteditable]')) return;
    if (event.key.toLowerCase() === 'p') {
        event.preventDefault();
        enterPresentation();
    } else if (startupFinished && overrideKeys.has(event.key)) {
        event.preventDefault();
        setClassOverride(overrideKeys.get(event.key));
    }
});

$('menuBtn').onclick = () => setMenu(!$('menu').classList.contains('open'));
document.addEventListener('click', event => {
    if (!event.target.closest('.menu') && $('menu').classList.contains('open')) setMenu(false);
});
$('startupContinue').onclick = finishSplash;
$('startIntro').onclick = startIntro;
setupScanner();
$('splashName').addEventListener('animationend', () => {
    animationFinished = true;
    updateStartup();
}, {once: true});
setInterval(poll, 350);
setInterval(frames, 65);
poll();
