#!/usr/bin/env python3
"""
Footer Component
Author information and branding
"""

import gradio as gr


def create_footer():
    """Create the application footer with author information."""
    
    footer_html = """
        <div class="footer-box" style="
            border-top: 2px solid #667eea;
            padding: 20px;
            margin-top: 30px;
            background: linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%);
            border-radius: 8px;
            text-align: center;
        ">
            <p style="font-size: 16px; color: #333; margin: 8px 0;">
                <strong>Author:</strong> Frederick Gyasi
            </p>
            <p style="font-size: 14px; color: #555; margin: 8px 0;">
                📧 <a href="mailto:gyasi@musc.edu" style="color: #667eea; text-decoration: none; font-weight: 500;">
                    gyasi@musc.edu
                </a>
            </p>
            <p style="font-size: 14px; color: #555; margin: 8px 0;">
                <strong>Medical University of South Carolina</strong>
            </p>
            <p style="font-size: 13px; color: #666; margin: 8px 0;">
                Biomedical Informatics Center | ClinicalNLP Lab
            </p>
            <p style="font-size: 12px; color: #999; margin: 15px 0 0 0;">
                © 2025 | Version 1.0.0 | MIT License
            </p>
        </div>
    """
    
    return gr.HTML(footer_html)