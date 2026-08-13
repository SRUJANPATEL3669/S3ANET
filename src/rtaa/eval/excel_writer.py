from __future__ import annotations

import openpyxl
from openpyxl.styles import PatternFill, Font
from pathlib import Path

def write_benchmark_results(results: list[dict], output_path: str | Path):
    """
    results: list of dicts, each containing:
    dataset, model, attack, OA, Kappa, AA, class_accs (list),
    train_time, test_time, total_time, SAM, SID, phys_consistency, ASR
    """
    wb = openpyxl.Workbook()
    # Remove default sheet
    wb.remove(wb.active)
    
    datasets = set(r["dataset"] for r in results)
    
    header = [
        "Model", "Attack", "OA (%)", "Kappa", "AA (%)", 
        "Train Time (s)", "Test Time (s)", "Total Time (s)", 
        "SAM (deg)", "SID", "Phys. Consistency (%)", "ASR (%)"
    ]
    
    for ds in datasets:
        ws = wb.create_sheet(title=ds)
        ds_results = [r for r in results if r["dataset"] == ds]
        
        if not ds_results:
            continue
            
        n_classes = len(ds_results[0].get("class_accs", []))
        class_headers = [f"Class {i+1} OA (%)" for i in range(n_classes)]
        
        full_header = header[:5] + class_headers + header[5:]
        ws.append(full_header)
        
        # Style header
        for cell in ws[1]:
            cell.font = Font(bold=True)
            
        for r in ds_results:
            row = [
                r.get("model", ""),
                r.get("attack", ""),
                r.get("OA", 0.0),
                r.get("Kappa", 0.0),
                r.get("AA", 0.0)
            ]
            row.extend(r.get("class_accs", [0.0] * n_classes))
            row.extend([
                r.get("train_time", 0.0),
                r.get("test_time", 0.0),
                r.get("total_time", 0.0),
                r.get("SAM", 0.0),
                r.get("SID", 0.0),
                r.get("phys_consistency", 0.0),
                r.get("ASR", 0.0)
            ])
            ws.append(row)
            
    # Add a summary sheet
    ws_sum = wb.create_sheet(title="Summary", index=0)
    ws_sum.append(["Dataset", "Model", "Attack", "OA (%)", "ASR (%)"])
    for cell in ws_sum[1]:
        cell.font = Font(bold=True)
        
    for r in results:
        ws_sum.append([
            r.get("dataset", ""),
            r.get("model", ""),
            r.get("attack", ""),
            r.get("OA", 0.0),
            r.get("ASR", 0.0)
        ])
        
    wb.save(output_path)
