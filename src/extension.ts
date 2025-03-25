// src/extension.ts
import * as vscode from 'vscode';
import { spawn } from 'child_process';
import * as path from 'path';
import * as fs from 'fs';

interface MergeConflict {
    originalCode: string | null;
    branchACode: string;
    branchBCode: string;
    context: string;
    filePath: string;
}

let activePanel: vscode.WebviewPanel | null = null;

export function activate(context: vscode.ExtensionContext) {
    const conflictStatusBarItem = vscode.window.createStatusBarItem(vscode.StatusBarAlignment.Left, 0);
    conflictStatusBarItem.text = "$(git-merge) Resolve Merge Conflicts";
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
        const conflicts: MergeConflict[] = parseConflicts(text, document.uri.fsPath);

        if (conflicts.length === 0) {
            vscode.window.showInformationMessage('No merge conflicts found');
            return;
        }

        vscode.window.showInformationMessage(`Found ${conflicts.length} merge conflict(s)`);

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
                showInitialWebview(conflict);
                const resolution = await resolveWithPython(conflict);
                if (resolution) {
                    if (activePanel) updateWebviewWithResolution(resolution, conflict);
                } else {
                    vscode.window.showErrorMessage('⚠️ Could not generate resolution');
                }
            }
        }
    });

    context.subscriptions.push(findConflictsCommand);
}

function parseConflicts(text: string, filePath: string): MergeConflict[] {
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
                filePath: filePath,
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

function showInitialWebview(conflict: MergeConflict) {
    if (activePanel) activePanel.dispose();
    activePanel = vscode.window.createWebviewPanel('mergeResolution', 'Merge Conflict Resolution', vscode.ViewColumn.Beside, { 
        enableScripts: true,
        retainContextWhenHidden: true 
    });
    activePanel.webview.html = getInitialWebviewContent();

    const panel = activePanel;
    panel.webview.onDidReceiveMessage(async (message) => {
        if (message.command === 'decline') {
            panel.dispose();
        } else if (message.command === 'accept') {
            try {
                // Replace file contents with the resolved code
                const edit = new vscode.WorkspaceEdit();
                const uri = vscode.Uri.file(conflict.filePath);
                edit.replace(
                    uri, 
                    new vscode.Range(
                        new vscode.Position(0, 0), 
                        new vscode.Position(Number.MAX_VALUE, Number.MAX_VALUE)
                    ), 
                    message.resolution
                );
                
                await vscode.workspace.applyEdit(edit);
                await vscode.commands.executeCommand('workbench.action.files.save');
                
                vscode.window.showInformationMessage('✅ Merge conflict resolved successfully!');
                panel.dispose();
            } catch (error) {
                vscode.window.showErrorMessage(`❌ Failed to apply resolution: ${error}`);
            }
        }
    });
}

function updateWebviewWithResolution(resolution: string, conflict: MergeConflict) {
    if (!activePanel) return;

    activePanel.webview.postMessage({ 
        command: 'updateResolution', 
        resolution: resolution,
        filepath: conflict.filePath
    });
}

function getInitialWebviewContent() {
    return `<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Merge Conflict Resolution</title>
    <style>
        :root {
            --vscode-editor-background: #1E1E1E;
            --vscode-editor-foreground: #D4D4D4;
            --vscode-editorLineNumber-foreground: #858585;
            --vscode-editor-selectionBackground: #264F78;
            --vscode-button-background: #0E639C;
            --vscode-button-hoverBackground: #1177BB;
            --vscode-button-foreground: white;
            --vscode-button-secondaryBackground: #3C3C3C;
            --vscode-button-secondaryHoverBackground: #4A4A4A;
        }
        body {
            font-family: 'Cascadia Code', 'Fira Code', monospace;
            background-color: var(--vscode-editor-background);
            color: var(--vscode-editor-foreground);
            padding: 20px;
            line-height: 1.6;
            margin: 0;
        }
        #output {
            background-color: #2C2C2C;
            border: 1px solid #3C3C3C;
            border-radius: 4px;
            padding: 15px;
            white-space: pre-wrap;
            font-size: 14px;
            max-height: 400px;
            overflow-y: auto;
            position: relative;
        }
        .button-container {
            display: flex;
            justify-content: flex-start;
            margin-top: 15px;
            gap: 10px;
        }
        .btn {
            padding: 8px 16px;
            border: none;
            border-radius: 4px;
            cursor: pointer;
            font-weight: 600;
            transition: background-color 0.2s;
        }
        .btn-accept {
            background-color: var(--vscode-button-background);
            color: var(--vscode-button-foreground);
        }
        .btn-accept:hover {
            background-color: var(--vscode-button-hoverBackground);
        }
        .btn-decline {
            background-color: var(--vscode-button-secondaryBackground);
            color: var(--vscode-button-foreground);
        }
        .btn-decline:hover {
            background-color: var(--vscode-button-secondaryHoverBackground);
        }
        .blinking-cursor {
            display: inline-block;
            animation: blink 1s steps(2, start) infinite;
            color: var(--vscode-editor-foreground);
        }
        @keyframes blink { to { visibility: hidden; } }
        #filepath {
            color: var(--vscode-editorLineNumber-foreground);
            margin-bottom: 10px;
            font-style: italic;
        }
    </style>
</head>
<body>
    <h3>🔧 Merge Conflict Resolution</h3>
    <div id="filepath">Resolving conflict...</div>
    <div id="output">
        <span style="color: #6A9955;">// Generating optimal merge resolution...</span>
        <span class="blinking-cursor">|</span>
    </div>
    <div class="button-container">
        <button class="btn btn-accept" id="acceptBtn" disabled onclick="accept()">✅ Accept</button>
        <button class="btn btn-decline" onclick="decline()">❌ Decline</button>
    </div>

    <script>
        const vscode = acquireVsCodeApi();
        let currentFilePath = '';

        window.addEventListener('message', event => {
            const message = event.data;
            if (message.command === 'updateResolution') {
                currentFilePath = message.filepath;
                document.getElementById('filepath').textContent = \`File: \${currentFilePath}\`;
                animateResolution(message.resolution);
            }
        });

        function animateResolution(text) {
            const outputDiv = document.getElementById("output");
            outputDiv.innerHTML = ''; // Clear previous content
            let i = 0;
            function animate() {
                if (i < text.length) {
                    outputDiv.textContent += text[i++];
                    setTimeout(animate, 20);
                } else {
                    outputDiv.innerHTML += '<span class="blinking-cursor">|</span>';
                    document.getElementById('acceptBtn').disabled = false;
                }
            }
            animate();
        }

        function accept() {
            const resolution = document.getElementById('output').textContent.replace(/\|$/, '').trim();
            vscode.postMessage({ 
                command: 'accept', 
                resolution: resolution 
            });
        }

        function decline() {
            vscode.postMessage({ command: 'decline' });
        }
    </script>
</body>
</html>`;
}

export function deactivate() {}