// Sign-in, first-admin setup and invite/reset acceptance. Tokens arrive in the URL
// fragment (#token=...), which browsers never send to the server or in a Referer header.
const $ = id => document.getElementById(id);
const mode = location.pathname.replace('/', '') || 'login';     // login | setup | accept
const token = new URLSearchParams(location.hash.slice(1)).get('token') || '';
history.replaceState(null, '', location.pathname);              // drop the token from the address bar

async function post(path, body) {
  const r = await fetch(path, { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(body) });
  const js = await r.json().catch(() => ({}));
  if (!r.ok) { const e = new Error(js.error || `Error ${r.status}`); e.needCode = js.need_code; throw e; }
  return js;
}

function show(id, on = true) { $(id).hidden = !on; }
function fail(msg) { $('err').textContent = msg; $('submit').disabled = false; }

async function init() {
  const st = await fetch('/api/auth/status').then(r => r.json()).catch(() => ({}));
  if (st.signed_in && mode === 'login') return location.replace('/');
  if (mode === 'setup') {
    if (!st.needs_setup) return location.replace('/login');
    $('title').textContent = 'Create the admin account';
    $('intro').textContent = 'You opened the one-time link from the server terminal. This account will be the only one that can invite people.';
    if (!token) {
      $('intro').textContent = 'Open the full one-time link printed in the server terminal (including the part after #), or run  python3 scripts/admin.py create  in the project folder.';
      show('form', false);
      return;
    }
    show('nameRow'); show('confirmRow'); show('hint');
    $('password').autocomplete = 'new-password';
    $('submit').textContent = 'Create admin account';
  } else if (mode === 'accept') {
    const info = token ? await fetch(`/api/auth/invite_info?token=${encodeURIComponent(token)}`).then(r => r.json()) : { valid: false };
    if (!info.valid) {
      $('title').textContent = 'Link not valid';
      $('intro').textContent = 'This link is invalid, already used, revoked or expired. Ask the admin for a new one.';
      show('form', false);
      return;
    }
    const reset = info.purpose === 'reset';
    $('title').textContent = reset ? 'Set a new password' : 'Accept your invitation';
    $('intro').textContent = reset ? `Choose a new password for ${info.email}. You will be signed out everywhere else.`
      : `You've been invited as ${info.email}. Choose a password to finish.`;
    $('email').value = info.email; $('email').readOnly = true;
    if (!reset) show('nameRow');
    show('confirmRow'); show('hint');
    $('password').autocomplete = 'new-password';
    $('submit').textContent = reset ? 'Save new password' : 'Create my account';
  }
}

$('form').onsubmit = async e => {
  e.preventDefault();
  $('err').textContent = '';
  const pw = $('password').value;
  if (mode !== 'login' && pw !== $('confirm').value) return fail('The two passwords are different.');
  $('submit').disabled = true;
  try {
    if (mode === 'login') await post('/api/auth/login', { email: $('email').value, password: pw, code: $('code').value });
    else if (mode === 'setup') await post('/api/auth/setup', { token, email: $('email').value, name: $('name').value, password: pw });
    else {
      const r = await post('/api/auth/accept', { token, name: $('name').value, password: pw });
      if (r.next) return location.replace(r.next);
    }
    location.replace('/');
  } catch (err) {
    if (err.needCode) { show('codeRow'); $('code').focus(); }
    fail(err.message);
  }
};

init();
