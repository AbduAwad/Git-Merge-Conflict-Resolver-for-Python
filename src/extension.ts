// src/extension.ts
import * as vscode from 'vscode';
import { spawn } from 'child_process';
import * as path from 'path';

interface MergeConflict {
    originalCode: string | null;
    branchACode: string;
    branchBCode: string;
    context: string;
}

let activePanel: vscode.WebviewPanel | null = null;

export function activate(context: vscode.ExtensionContext) {
    const conflictStatusBarItem = vscode.window.createStatusBarItem(vscode.StatusBarAlignment.Right, 100);
    conflictStatusBarItem.text = "$(git-merge) Find Conflicts";
    conflictStatusBarItem.tooltip = "Find Merge Conflicts in Current File";
    conflictStatusBarItem.command = 'mergeConflictReader.findConflicts';
    conflictStatusBarItem.show();
    context.subscriptions.push(conflictStatusBarItem);

    const findConflictsCommand = vscode.commands.registerCommand('mergeConflictReader.findConflicts', async () => {
        const editor = vscode.window.activeTextEditor;
        if (!editor) {
            vscode.window.showErrorMessage('No active text editor');
            return;
        }

        const document = editor.document;
        const text = document.getText();
        const conflicts: MergeConflict[] = parseConflicts(text);

        if (conflicts.length === 0) {
            vscode.window.showInformationMessage('No merge conflicts found');
            return;
        }

        const outputChannel = vscode.window.createOutputChannel("Merge Conflicts");
        outputChannel.clear();
        outputChannel.show(true);

        conflicts.forEach((conflict, index) => {
            outputChannel.appendLine(`Conflict ${index + 1}:`);
            outputChannel.appendLine('Branch A Code:');
            outputChannel.appendLine(conflict.branchACode);
            outputChannel.appendLine('\nBranch B Code:');
            outputChannel.appendLine(conflict.branchBCode);
            if (conflict.originalCode) {
                outputChannel.appendLine('\nOriginal Code:');
                outputChannel.appendLine(conflict.originalCode);
            }
            outputChannel.appendLine('-'.repeat(50));
        });

        vscode.window.showInformationMessage(`Found ${conflicts.length} merge conflict(s). Check the OUTPUT panel.`);

        const modelOptions = [
            {
                label: "🧠 CodeT5",
                description: "Fine-tuned Transformer",
                detail: "Generates a custom resolution using a fine-tuned CodeT5 model trained on thousands of realistic merge conflicts.",
                value: "codet5"
            },
            {
                label: "🧪 RNN",
                description: "Binary Classifier",
                detail: "Classifies the better branch and selects it as the resolved output.",
                value: "custom"
            }
        ];

        const selected = await vscode.window.showQuickPick(modelOptions, {
            placeHolder: "Select a model to resolve merge conflicts"
        });

        if (!selected) return;
        vscode.window.showInformationMessage(`✅ Selected model: ${selected.label}`);

        if (selected.value === "codet5") {
            for (const conflict of conflicts) {
                showLoadingWebview();
                const resolution = await resolveWithPython(conflict);
                if (resolution) {
                    if (activePanel) activePanel.dispose();
                    showWebviewPanel(resolution);
                } else {
                    vscode.window.showErrorMessage('⚠️ Could not generate resolution');
                }
            }
        }
    });

    context.subscriptions.push(findConflictsCommand);
}

function parseConflicts(text: string): MergeConflict[] {
    const conflicts: MergeConflict[] = [];
    const lines = text.split('\n');
    let branchA: string[] = [], branchB: string[] = [], i = 0;

    while (i < lines.length) {
        if (lines[i].startsWith('<<<<<<< HEAD')) {
            const beforeConflict = lines.slice(0, i);
            i++;
            const branchALines: string[] = [];
            while (i < lines.length && !lines[i].startsWith('=======')) branchALines.push(lines[i++]);
            i++;
            const branchBLines: string[] = [];
            while (i < lines.length && !lines[i].startsWith('>>>>>>>')) branchBLines.push(lines[i++]);
            i++;
            const afterConflict = lines.slice(i);

            branchA = [...beforeConflict, ...branchALines, ...afterConflict];
            branchB = [...beforeConflict, ...branchBLines, ...afterConflict];

            conflicts.push({
                context: text,
                originalCode: branchA.join('\n').trim(),
                branchACode: branchA.join('\n').trim(),
                branchBCode: branchB.join('\n').trim()
            });
        } else {
            i++;
        }
    }
    return conflicts;
}

function resolveWithPython(conflict: MergeConflict): Promise<string> {
    return new Promise((resolve, reject) => {
        const scriptPath = path.join(__dirname, '..', 'transformer_model', 'resolve_conflict.py');
        const input = JSON.stringify({
            original: conflict.originalCode ?? '',
            branchA: conflict.branchACode,
            branchB: conflict.branchBCode
        });

        const pythonPath = `C:\\Users\\sheri\\OneDrive - Carleton University\\Comp_Courses\\Comp4107\\FINAL_PROJECT\\COMP4107_FP\\myenv\\Scripts\\python.exe`;
        const child = spawn(pythonPath, [scriptPath]);

        let output = '';
        let error = '';

        child.stdout.on('data', data => output += data);
        child.stderr.on('data', data => error += data);
        child.on('close', code => {
            if (code !== 0 || error) {
                vscode.window.showErrorMessage(`❌ Python error: ${error}`);
                return reject(error);
            }
            resolve(output.trim());
        });

        child.stdin.write(input);
        child.stdin.end();
    });
}

function showLoadingWebview() {
    if (activePanel) activePanel.dispose();
    activePanel = vscode.window.createWebviewPanel('mergeResolution', 'Loading Resolution...', vscode.ViewColumn.Beside, { enableScripts: true });
    activePanel.webview.html = `<!DOCTYPE html>
    <html><head><style>
    body { font-family: monospace; background: #0d1117; color: #c9d1d9; display: flex; justify-content: center; align-items: center; height: 100vh; }
    .loader { border: 6px solid #161b22; border-top: 6px solid #58a6ff; border-radius: 50%; width: 50px; height: 50px; animation: spin 1s linear infinite; }
    @keyframes spin { 100% { transform: rotate(360deg); } }
    </style></head><body>
    <div class="loader"></div>
    </body></html>`;
}

function showWebviewPanel(resolution: string) {
    if (activePanel) activePanel.dispose();
    activePanel = vscode.window.createWebviewPanel('mergeResolution', 'Merge Conflict Resolution', vscode.ViewColumn.Beside, { enableScripts: true });
    activePanel.webview.html = getWebviewContent(resolution);

    const panel = activePanel;
    panel.webview.onDidReceiveMessage(message => {
        const editor = vscode.window.activeTextEditor;
        if (!editor) return;
        if (message.command === 'accept') {
            editor.edit(editBuilder => {
                const start = new vscode.Position(0, 0);
                const end = new vscode.Position(editor.document.lineCount + 1, 0);
                editBuilder.delete(new vscode.Range(start, end));
                editBuilder.insert(new vscode.Position(0, 0), message.resolution);
            });
            panel.dispose();
        } else if (message.command === 'decline') {
            panel.dispose();
        }
    });
}

function getWebviewContent(resolution: string) {
    return `<!DOCTYPE html>
<html lang="en">
<head>
    <style>
        body { font-family: monospace; padding: 20px; background: #0d1117; color: #c9d1d9; }
        #output { white-space: pre-wrap; border: 1px solid #30363d; padding: 15px; background: #161b22; min-height: 200px; border-radius: 6px; }
        .blinking-cursor { display: inline-block; animation: blink 1s steps(2, start) infinite; color: #58a6ff; }
        @keyframes blink { to { visibility: hidden; } }
        button {
            margin-top: 12px;
            margin-right: 8px;
            padding: 6px 12px;
            background: #238636;
            color: white;
            border: none;
            border-radius: 4px;
            cursor: pointer;
        }
        button.decline { background: #da3633; }
    </style>
</head>
<body>
    <h3>🔧 Suggested Merge Resolution</h3>
    <div id="output"></div><span class="blinking-cursor">|</span><br/>
    <button onclick="accept()">✅ Accept</button>
    <button class="decline" onclick="decline()">❌ Decline</button>

    <script>
        const vscode = acquireVsCodeApi();
        const text = ${JSON.stringify(resolution)};
        const outputDiv = document.getElementById("output");
        let i = 0;
        function animate() {
            if (i < text.length) {
                outputDiv.textContent += text[i++];
                setTimeout(animate, 40);
            }
        }
        animate();

        function accept() {
            vscode.postMessage({ command: 'accept', resolution: text });
        }
        function decline() {
            vscode.postMessage({ command: 'decline' });
        }
    </script>
</body>
</html>`;
}

export function deactivate() {}
