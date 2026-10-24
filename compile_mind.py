"""Compile l'image de la fiche en cible MindAR (models/fiche.mind) dans un Chromium sans écran.

Usage : python3 compile_mind.py site
Nécessite : site/lib/mindar/ (MindAR 1.2.5) et site/models/fiche-cible.png ; pip install playwright.
"""
import asyncio, base64, functools, http.server, pathlib, socketserver, sys, threading
from playwright.async_api import async_playwright

SITE = pathlib.Path(sys.argv[1]).resolve()
class Quiet(http.server.SimpleHTTPRequestHandler):
    def log_message(self, *a): pass
socketserver.TCPServer.allow_reuse_address = True
srv = socketserver.ThreadingTCPServer(('127.0.0.1', 0), functools.partial(Quiet, directory=str(SITE)))
port = srv.server_address[1]
threading.Thread(target=srv.serve_forever, daemon=True).start()

JS = """async () => {
  const { Compiler } = await import('/lib/mindar/mindar-image.prod.js');
  const img = new Image(); img.src = '/models/fiche-cible.png';
  await img.decode();
  const c = new Compiler();
  await c.compileImageTargets([img], () => {});
  const buf = await c.exportData();
  let s = ''; const b = new Uint8Array(buf);
  for (let i = 0; i < b.length; i += 0x8000) s += String.fromCharCode.apply(null, b.subarray(i, i + 0x8000));
  return btoa(s);
}"""

async def main():
    async with async_playwright() as p:
        b = await p.chromium.launch(args=['--use-gl=angle', '--use-angle=swiftshader', '--enable-unsafe-swiftshader'])
        pg = await b.new_page()
        await pg.goto(f'http://127.0.0.1:{port}/models/fiche-cible.png')
        data = await pg.evaluate(JS)
        out = SITE / 'models' / 'fiche.mind'
        out.write_bytes(base64.b64decode(data))
        print('cible compilée :', out, out.stat().st_size, 'octets')
        await b.close()

asyncio.run(main())
