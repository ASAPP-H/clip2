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

label2abbrev = {'I-Imaging-related followup': 'imaging',
        'I-Appointment-related followup': 'appointment',
        'I-Medication-related followups': 'medication',
        'I-Procedure-related followup': 'procedure',
        'I-Lab-related followup': 'lab',
        'I-Case-specific instructions for patient': 'case',
        'I-Other helpful contextual information': 'other',
 }

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
PWD = os.environ.get('CLIP_DIR')
PAD = "__PAD__"
UNK = "__UNK__"
