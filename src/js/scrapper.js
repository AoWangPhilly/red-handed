"use strict";
const puppeteer = require("puppeteer");

const scrapeNews = async url => {
    const browser = await puppeteer.launch();
    const page = await browser.newPage();
    await page.goto(url);

    const [el] = await page.$x('//*[@id="fitt-analytics"]/div/main/div[2]/div[1]/section/div/a/div[1]/img');
    const src = await el.getProperty("src");
    const srcTxt = await src.jsonValue();

    const [el2] = await page.$x('//*[@id="fitt-analytics"]/div/main/div[2]/div[1]/section/div/a/div[2]');
    const txt = await el2.getProperty("textContent");
    const rawTxt = await txt.jsonValue();

    const [el3] = await page.$x('//*[@id="fitt-analytics"]/div/main/div[2]/div[1]/section/div/a');
    const href = await el3.getProperty("href");
    const hrefTxt = await href.jsonValue();

    await browser.close();
    return {srcTxt, rawTxt, hrefTxt};
}

let x = scrapeNews("https://6abc.com/tag/crime/");
(async () => {
    let [link, h1, href] = (await x);
    console.log(link);
})()