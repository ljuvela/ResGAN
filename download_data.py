import requests

# Adapted from https://stackoverflow.com/a/39225039

def download_file_from_google_drive(id, destination):
    def get_confirm_token(response):
        for key, value in response.cookies.items():
            if key.startswith('download_warning'):
                return value

        return None

    def save_response_content(response, destination):
        CHUNK_SIZE = 32768

        with open(destination, "wb") as f:
            for chunk in response.iter_content(CHUNK_SIZE):
                if chunk: # filter out keep-alive new chunks
                    f.write(chunk)

    URL = "https://docs.google.com/uc?export=download"

    session = requests.Session()

    response = session.get(URL, params = { 'id' : id }, stream = True)
    token = get_confirm_token(response)

    if token:
        params = { 'id' : id, 'confirm' : token }
        response = session.get(URL, params = params, stream = True)

    save_response_content(response, destination)

if __name__ == "__main__":
    import sys
 
    file_id = '1i3FUu3V87YaYKZvXQZ2VxJ0s_iXuQM4S'
    # DESTINATION FILE ON YOUR DISK
    destination = './data.zip'
    download_file_from_google_drive(file_id, destination)        
