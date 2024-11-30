const { app, desktopCapturer, screen } = require('electron');
const WebSocket = require('ws');

let ws;

// 启用 GPU 捕获支持
app.commandLine.appendSwitch('disable-features', 'UseModernGPUVideoCapture');
app.commandLine.appendSwitch('enable-usermedia-screen-capturing');

app.on('ready', async () => {
    console.log('Electron 应用已启动');

    // 设置缩放比例为 1.0
    const { webContents } = require('electron');
    webContents.getAllWebContents().forEach((content) => {
        content.setZoomFactor(1.0);
    });

    // 初始化 WebSocket
    initWebSocket();

    // 开始屏幕捕获
    startScreenCapture();
});

/**
 * 初始化 WebSocket 连接
 */
function initWebSocket() {
    ws = new WebSocket('ws://127.0.0.1:8765');

    ws.on('open', () => {
        console.log('WebSocket 已连接');
    });

    ws.on('message', (message) => {
        console.log('从 Python 收到消息:', message);
    });

    ws.on('error', (err) => {
        console.error('WebSocket 错误:', err);
    });

    ws.on('close', () => {
        console.log('WebSocket 已关闭');
    });
}

/**
 * 捕获整个屏幕内容
 */
async function startScreenCapture() {
    // 动态获取主屏幕分辨率
    const primaryDisplay = screen.getPrimaryDisplay();
    const { width, height } = primaryDisplay.size;
    console.log(`主屏幕分辨率: ${width}x${height}`);

    // 获取屏幕捕获源
    const sources = await desktopCapturer.getSources({
        types: ['screen'],
        thumbnailSize: { width, height }, // 设置捕获分辨率为屏幕分辨率
    });

    const screenSource = sources[0];
    if (!screenSource) {
        console.error('未找到屏幕源');
        return;
    }

    console.log('捕获屏幕:', screenSource.name);

    // 循环捕获截图并发送
    setInterval(() => {
        try {
            const screenshotBase64 = screenSource.thumbnail.toDataURL().replace(/^data:image\/png;base64,/, '');

            // 发送屏幕截图
            if (ws && ws.readyState === WebSocket.OPEN) {
                ws.send(screenshotBase64);
                console.log('屏幕截图已发送到 Python');
            }
        } catch (error) {
            console.error('截图发送失败:', error);
        }
    }, 500); // 每 500 毫秒捕获一次
}
