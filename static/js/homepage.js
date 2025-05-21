const sources = Array.from(document.querySelectorAll('.intro .source'))
    .map(p => p.textContent);
const el = document.getElementById('typewriter');
let idx = 0;

function startTyping() {
    const text = sources[idx];
    // set CSS vars so the animation steps and width match this text
    el.style.setProperty('--length', text.length + 'ch');
    el.style.setProperty('--chars', text.length);
    el.textContent = text;

    // trigger the animation class
    el.classList.add('typing');
}

// when the typing animation finishes, queue the next one
el.addEventListener('animationend', e => {
    if (e.animationName === 'typing') {
        // remove the class so we can re‑add it next time
        el.classList.remove('typing');
        // advance index (loop back around)
        idx = (idx + 1) % sources.length;
        // small pause before next loop
        setTimeout(startTyping, 2000);
    }
});

// kick it off
startTyping();