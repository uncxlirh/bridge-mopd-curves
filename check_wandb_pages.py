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
                    assert '六领域正式评测' in body, url
                    if '/reports/' in path:
                        assert all(t in body for t in ['为什么看这些指标','正向结果','负向结果','基本持平的结果','为什么做教师切换和归一化']), url
                        assert '47.04' in body and '61.90' in body and '18.45' in body, url
                        assert '94/100' not in body and '等待完整评测' not in body, url
                    else:
                        assert '六域均分增益' in body and '最差领域恢复率增益' in body, url
                    i=len(list(dest.glob('*.json')))
                    (dest/f'{i}.json').write_text(json.dumps(info,ensure_ascii=False,indent=2))
                    reject = page.get_by_role('button',name='Reject All',exact=True)
                    if await reject.count():
                        await reject.first.click(timeout=5000)
                    await page.screenshot(path=str(dest/f'{i}.png'),full_page=False)
                    if '/reports/' in path:
                        for j,title in enumerate(['六领域正式评测结果','恢复率和最差领域']):
                            await page.get_by_text(title,exact=True).scroll_into_view_if_needed()
                            await page.wait_for_timeout(8000)
                            if title == '六领域正式评测结果':
                                await page.get_by_text('100 步 · 相对普通 MOPD 的各域增减',exact=True).first.scroll_into_view_if_needed()
                                await page.mouse.wheel(0,240)
                                await page.wait_for_timeout(4000)
                            await page.screenshot(path=str(dest/f'{i}-section-{j}.png'),full_page=False)
                            section_text=await page.locator('body').inner_text()
                            assert 'Page not found' not in section_text
                            print('RENDERED REPORT SECTION',title,flush=True)

                except Exception as e:
                    print(type(e).__name__,str(e)[:500],flush=True)
                    raise
        await browser.close()
asyncio.run(main())
