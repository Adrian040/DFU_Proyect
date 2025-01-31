import csv

def append_csv(input_path, output_path):
    # Leer el archivo CSV de entrada
    with open(input_path, mode='r', newline='') as infile:
        reader = csv.reader(infile)
        headers = next(reader)
        values = next(reader)
    
    # Leer el archivo CSV de salida y verificar si tiene encabezados
    try:
        with open(output_path, mode='r', newline='') as outfile:
            reader = csv.reader(outfile)
            existing_headers = next(reader)
    except FileNotFoundError:
        existing_headers = []

    # Si el archivo de salida no tiene encabezados, escribirlos
    if not existing_headers:
        with open(output_path, mode='w', newline='') as outfile:
            writer = csv.writer(outfile)
            writer.writerow(headers)
            writer.writerow(values)
    else:
        # Si el archivo de salida ya tiene encabezados, agregar los nuevos valores
        with open(output_path, mode='a', newline='') as outfile:
            writer = csv.writer(outfile)
            writer.writerow(values)

inp_pth_params = "C:/Users/am969/Documents/DFU_Proyect/SegmentationNetworks/Models/ResUnet/output_assets_model/parameters_ResUnet.csv"
out_pth_params = "C:/Users/am969/Documents/DFU_Proyect/SegmentationNetworks/trained_Models/DataModels.csv"

append_csv(inp_pth_params, out_pth_params)