import http from "./comercialhttp";
import axios from "axios";

const CommercialGetColumn = async (
  layout_file: File[] | null,
  api_Message: (messages: string[]) => void
) => {
  const formData = new FormData();

  if (layout_file && layout_file.length > 0) {
    formData.append("file", layout_file[0]); // Assuming single file upload
  }

  try {
    const response = await http.post("/find_similar_columns", formData, {
      headers: {
        "Content-Type": "multipart/form-data",
        "Access-Control-Allow-Origin": "*",
        Accept: "application/json",
      },
    });

    // if (response.status !== 200) {
    //   throw new Error(`Failed to trigger /process. Status: ${response.status}`);
    // }
    api_Message(["column extracted successfully", "#21a67a"]);
    setTimeout(() => {
      api_Message([]);
    }, 3000);
    return response; // Return data or handle as needed
  } catch (error: any) {
    if (axios.isAxiosError(error)) {
      console.error("Axios error:", error.response?.data?.error);
      api_Message([error.response?.data?.error, "#EF3934"]);
      setTimeout(() => {
        api_Message([]);
      }, 3000);
    } else {
      console.error("Error uploading or processing files:", error.message);
      api_Message([error.message, "#EF3934"]);
      setTimeout(() => {
        api_Message([]);
      }, 3000);
    }
    console.error("Error uploading files:", error.message);
    throw error; // Throw or handle error as needed
  }
};

export default CommercialGetColumn;
