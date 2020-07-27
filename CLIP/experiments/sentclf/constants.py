import os

import torch

LABEL_TYPES = ['I-Imaging-related followup',
 'I-Appointment-related followup',
 'I-Medication-related followups',
 'I-Procedure-related followup',
 'I-Lab-related followup',
 'I-Case-specific instructions for patient',
 'I-Other helpful contextual information',
 ]

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
PWD = os.environ.get('CLIP_DIR')
PAD = "__PAD__"
UNK = "__UNK__"
