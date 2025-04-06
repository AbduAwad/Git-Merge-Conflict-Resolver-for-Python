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

interface RnnResult {
    selected_branch: "A" | "B";
    confidence: number;
    resolved_code: string;
    error?: string;
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
                value: "rnn"
            }
        ];

        const selected = await vscode.window.showQuickPick(modelOptions, {
            placeHolder: "Select a model to resolve merge conflicts"
        });

        if (!selected) return;
        vscode.window.showInformationMessage(`Selected model: ${selected.label}`);

        if (selected.value === "codet5") {
            for (const conflict of conflicts) {
                showInitialWebview(conflict, false);
                const resolution = await resolveWithPython(conflict, 'codet5');
                if (resolution) {
                    if (activePanel) updateWebviewWithResolution(resolution, conflict);
                } else {
                    vscode.window.showErrorMessage('> Could not generate resolution');
                }
            }
        } else if (selected.value === "rnn") {
            for (const conflict of conflicts) {
                showInitialWebview(conflict, true);
                const result = await resolveWithPython(conflict, 'rnn');
                if (result) {
                    try {
                        const rnnResult = JSON.parse(result) as RnnResult;
                        if (rnnResult.error) {
                            vscode.window.showErrorMessage(`RNN model error: ${rnnResult.error}`);
                        }
                        if (activePanel) updateWebviewWithRnnResolution(rnnResult, conflict);
                    } catch (e) {
                        vscode.window.showErrorMessage('> Could not parse RNN model result');
                    }
                } else {
                    vscode.window.showErrorMessage('> Could not classify conflict');
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

function resolveWithPython(conflict: MergeConflict, modelType: 'codet5' | 'rnn'): Promise<string> {
    return new Promise((resolve, reject) => {

        // Choose the appropriate script based on the model type
        const scriptName = modelType === 'codet5' ? 'resolve_conflict.py' : 'rnn_classifier.py';
        
        // Update this path to match your actual directory structure
        const scriptPath = modelType === 'codet5' 
            ? path.join(__dirname, '..', 'transformer_model', scriptName)
            : path.join(__dirname, '..', 'rnn_model', scriptName);

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
            try {
                const cleaned = output.trim();
            
                if (modelType === 'rnn') {
                    // RNN returns JSON
                    if (!cleaned.startsWith("{") || !cleaned.endsWith("}")) {
                        throw new Error("Expected JSON output but got plain text");
                    }
                    resolve(cleaned);
                } else {
                    resolve(cleaned); // CodeT5 returns plain text
                }
            } catch (err) {
                vscode.window.showErrorMessage(`❌ Failed to parse Python output:\n${output}\n\n⚠️ stderr:\n${error}`);
                reject(error || 'Invalid output');
            } 
        });
        child.stdin.write(input);
        child.stdin.end();
    });
}

function showInitialWebview(conflict: MergeConflict, isRnn: boolean = false) {
    if (activePanel) activePanel.dispose();
    activePanel = vscode.window.createWebviewPanel('mergeResolution', 'Merge Conflict Resolution', vscode.ViewColumn.Beside, { 
        enableScripts: true,
        retainContextWhenHidden: true 
    });
    activePanel.webview.html = getInitialWebviewContent(isRnn);

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

function updateWebviewWithRnnResolution(result: RnnResult, conflict: MergeConflict) {
    if (!activePanel) return;

    activePanel.webview.postMessage({ 
        command: 'updateRnnResolution', 
        result: result,
        filepath: conflict.filePath
    });
}

function getInitialWebviewContent(isRnn: boolean = false) {
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
        .branch-info {
            background-color: #3C3C3C;
            border-radius: 4px;
            padding: 8px 12px;
            margin-bottom: 15px;
            display: ${isRnn ? 'flex' : 'none'};
            align-items: center;
            gap: 10px;
        }
        .branch-badge {
            border-radius: 4px;
            padding: 4px 8px;
            font-weight: bold;
        }
        .branch-a {
            background-color: #3498db;
            color: white;
        }
        .branch-b {
            background-color: #9b59b6;
            color: white;
        }
        .confidence {
            margin-left: auto;
            font-style: italic;
            color: #AAA;
        }
    </style>
</head>
<body>
    <h3>${isRnn ? '🧪 RNN Branch Classifier' : '🔧 Merge Conflict Resolution'}</h3>
    <div id="filepath">Resolving conflict...</div>
    
    <div class="branch-info" id="branchInfo">
        <span>Selected:</span>
        <span class="branch-badge" id="branchBadge">Branch ?</span>
        <span class="confidence" id="confidence">Confidence: 0%</span>
    </div>
    
    <div id="output">
        <span style="color: #6A9955;">${isRnn ? '// Classifying branches...' : '// Generating optimal merge resolution...'}</span>
        <span class="blinking-cursor">|</span>
    </div>
    <div class="button-container">
        <button class="btn btn-accept" id="acceptBtn" disabled onclick="accept()">✅ Accept</button>
        <button class="btn btn-decline" onclick="decline()">❌ Decline</button>
    </div>

    <script>
        const vscode = acquireVsCodeApi();
        let currentFilePath = '';
        let isRnn = ${isRnn};

        window.addEventListener('message', event => {
            const message = event.data;
            if (message.command === 'updateResolution') {
                currentFilePath = message.filepath;
                document.getElementById('filepath').textContent = \`File: \${currentFilePath}\`;
                animateResolution(message.resolution);
            } else if (message.command === 'updateRnnResolution') {
                currentFilePath = message.filepath;
                document.getElementById('filepath').textContent = \`File: \${currentFilePath}\`;
                
                // Update the branch info section
                const result = message.result;
                const branchBadge = document.getElementById('branchBadge');
                branchBadge.textContent = \`Branch \${result.selected_branch}\`;
                branchBadge.className = \`branch-badge branch-\${result.selected_branch.toLowerCase()}\`;
                
                // Update confidence
                document.getElementById('confidence').textContent = \`Confidence: \${Math.round(result.confidence * 100)}%\`;
                
                // Show the resolution
                animateResolution(result.resolved_code);
            }
        });

        function animateResolution(text) {
            const outputDiv = document.getElementById("output");
            outputDiv.innerHTML = ''; // Clear previous content
            let i = 0;

            const cursorSpan = document.createElement("span");
            cursorSpan.className = "blinking-cursor";
            cursorSpan.textContent = "|";
            outputDiv.appendChild(cursorSpan);

            function animate() {
                if (i < text.length) {
                    cursorSpan.insertAdjacentText('beforebegin', text[i++]);
                    setTimeout(animate, 10); // typing speed restored
                } else {
                    cursorSpan.remove(); // ✅ Remove the blinking cursor from DOM
                    document.getElementById('acceptBtn').disabled = false;
                }
            }
            animate();
        }



        function accept() {
            const outputDiv = document.getElementById('output');
            const resolution = Array.from(outputDiv.childNodes)
                .filter(node => node.nodeType === Node.TEXT_NODE)
                .map(node => node.textContent)
                .join('')
                .trim();

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