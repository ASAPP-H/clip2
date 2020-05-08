for label_type in "Procedure-related followup" "Imaging-related followup" 'Case-specific instructions for patient', 'Medication-related followups', 'Appointment-related followup', 'Other helpful contextual information', 'Lab-related followup'
do
        lr=0.005
        epochs=50
        binary_checkpoint="outputs_binary/binary0.005/models/master.pt"
        python main_finegrained.py "${lr}" "${epochs}" "${binary_checkpoint}" "${label_type}"
done
