"""Check the rendered, anonymous report: data, compact layout and best-score emphasis."""
import asyncio,json
from pathlib import Path
from playwright.async_api import async_playwright

async def main():
    out=Path('browser-pages');out.mkdir(exist_ok=True)
    async with async_playwright() as p:
        browser=await p.chromium.launch()
        page=await browser.new_page(viewport={'width':1440,'height':1000})
        root='https://wandb.ai/chenxheee-sjtu-hpc-center/bridge_mopd'
        for i,url in enumerate([root+'/reports/渐进式教师实验总览--VmlldzoxODMxNzAwNA==',root+'?nw=xmjip17v9ke']):
            await page.goto(url,wait_until='domcontentloaded',timeout=60000)
            await page.wait_for_timeout(12000)
            body=await page.locator('body').inner_text()
            assert 'Page not found' not in body and 'We couldn’t find this page' not in body
            assert 'Select runs that logged' not in body and 'No data available' not in body
            reject=page.get_by_role('button',name='Reject All',exact=True)
            if await reject.count():await reject.first.click()
            if i==0:
                assert all(t in body for t in ['总览','评测','设置与算法','领域变化','完整分项成绩','学生训练设置','5394','16384','96','47.04','61.90','18.45'])
                assert await page.locator('strong,b').count()>=10
                await page.screenshot(path=str(out/'report-top.png'))
                await page.get_by_text('阶段对比',exact=True).first.scroll_into_view_if_needed()
                await page.mouse.wheel(0,100)
                await page.wait_for_timeout(10000)
                body=await page.locator('body').inner_text()
                assert 'Select runs that logged' not in body and 'No data available' not in body
                await page.screenshot(path=str(out/'report-charts.png'))
                await page.get_by_text('领域变化',exact=True).first.scroll_into_view_if_needed()
                await page.screenshot(path=str(out/'report-domains.png'))
                print('RENDERED MiMo-style sections: pinned student metrics, separate evaluation, full student settings, score highlights',flush=True)
            else:
                # Workspaces virtualize sections outside the viewport. Inspect as we scroll.
                required=['阶段对比','学生训练','各领域蒸馏信号','教师进度','DN 各领域乘子']
                seen=body
                await page.mouse.move(1380,700)
                for _ in range(6):
                    if all(t in seen for t in required):break
                    await page.mouse.wheel(0,900)
                    await page.wait_for_timeout(1000)
                    seen+='\n'+await page.locator('body').inner_text()
                assert all(t in seen for t in required), seen[-9000:]
                body=seen
                await page.screenshot(path=str(out/'workspace.png'))
            (out/f'{i}.json').write_text(json.dumps({'url':page.url,'body':body},ensure_ascii=False,indent=2))
        await browser.close()
asyncio.run(main())
