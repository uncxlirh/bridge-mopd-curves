"""Inspect anonymous browser navigation, not just the metrics API."""
import asyncio,json
from pathlib import Path
from playwright.async_api import async_playwright

async def main():
    dest=Path('browser-pages');dest.mkdir(exist_ok=True)
    async with async_playwright() as p:
        browser=await p.chromium.launch()
        page=await browser.new_page(viewport={'width':1440,'height':1000})
        base='/chenxheee-sjtu-hpc-center/bridge_mopd'
        paths=[base+'/reports/渐进式教师实验总览--VmlldzoxODMxNzAwNA==',
               base+'?nw=xmjip17v9ke']
        for host in ['https://wandb.ai']:
            for path in paths:
                url=host+path
                try:
                    response=await page.goto(url,wait_until='domcontentloaded',timeout=60000)
                    await page.wait_for_timeout(12000)
                    body=(await page.locator('body').inner_text())[:14000]
                    info={'input':url,'url':page.url,'status':response.status if response else None,'title':await page.title(),'body':body}
                    print(json.dumps(info,ensure_ascii=False),flush=True)
                    assert 'Page not found' not in body and 'We couldn’t find this page' not in body, url
                    assert '六领域正式评测' in body and '1.7B 普通 MOPD' in body, url
                    i=len(list(dest.glob('*.json')))
                    (dest/f'{i}.json').write_text(json.dumps(info,ensure_ascii=False,indent=2))
                    await page.screenshot(path=str(dest/f'{i}.png'),full_page=False)
                except Exception as e:
                    print(type(e).__name__,str(e)[:500],flush=True)
                    raise
        await browser.close()
asyncio.run(main())
