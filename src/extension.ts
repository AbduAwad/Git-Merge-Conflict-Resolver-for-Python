// src/extension.ts
import * as vscode from 'vscode';

interface MergeConflict {
    originalCode: string | null;
    branchACode: string;
    branchBCode: string;
    context: string;
}

export function activate(context: vscode.ExtensionContext) {
    // Create status bar item
    const conflictStatusBarItem = vscode.window.createStatusBarItem(vscode.StatusBarAlignment.Right, 100);
    conflictStatusBarItem.text = "$(git-merge) Find Conflicts";
    conflictStatusBarItem.tooltip = "Find Merge Conflicts in Current File";
    conflictStatusBarItem.command = 'mergeConflictReader.findConflicts';
    conflictStatusBarItem.show();
    context.subscriptions.push(conflictStatusBarItem);

    // Register command to find conflicts
    const findConflictsCommand = vscode.commands.registerCommand('mergeConflictReader.findConflicts', () => {
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

        // Create an output channel to display conflicts
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
    });

    context.subscriptions.push(findConflictsCommand);
}


function parseConflicts(text: string): MergeConflict[] {
    const conflicts: MergeConflict[] = [];
    const lines = text.split('\n');

    let currentConflictStart = -1;
    let branchA: string[] = [];
    let branchB: string[] = [];
    let i = 0;

    while (i < lines.length) {
        if (lines[i].startsWith('<<<<<<< HEAD')) {
            currentConflictStart = i;

            const beforeConflict = lines.slice(0, i);

            // Collect branch A lines
            i++;
            const branchALines: string[] = [];
            while (i < lines.length && !lines[i].startsWith('=======')) {
                branchALines.push(lines[i]);
                i++;
            }

            i++; // Skip '=======' line

            // Collect branch B lines
            const branchBLines: string[] = [];
            while (i < lines.length && !lines[i].startsWith('>>>>>>>')) {
                branchBLines.push(lines[i]);
                i++;
            }

            i++; // Skip '>>>>>>> ...' line

            const afterConflict = lines.slice(i); // All remaining lines after the conflict

            branchA = [
                ...beforeConflict,
                ...branchALines,
                ...afterConflict
            ];

            branchB = [
                ...beforeConflict,
                ...branchBLines,
                ...afterConflict
            ];

            conflicts.push({
                context: text, // optionally: beforeConflict.join('\n') + '\n...\n' + afterConflict.join('\n'),
                originalCode: branchA.join('\n').trim(),
                branchACode: branchA.join('\n').trim(),
                branchBCode: branchB.join('\n').trim()
            });

            // Don't increment `i` here — already advanced during conflict collection
        } else {
            i++;
        }
    }

    return conflicts;
}

export function deactivate() {}