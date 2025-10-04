import http from "./facilitieshttp";
import axios from "axios";

const FacilitiesMultiplePostFile = async (
  // layout_file: File[] | null
  // selected_columns: string[],
  layout_file: File[] | null,
  commercial_file: File[] | null,
  api_Message: (messages: string[]) => void,
  selectedValue: string
  // fileColumns: { [key: string]: string[] },
  // selectedValue: string
) => {
  const formData = new FormData();
  if (selectedValue) {
    formData.append("choice", selectedValue);
  }
  // formData.append("choice", "without_format");
  // if (selectedValue) {
  //   formData.append("choice", selectedValue);
  // }

  // if (layout_file && layout_file.length > 0) {
  //   formData.append("file", layout_file[0]); // Assuming single file upload
  // }

  // Append each item in selected_columns as a separate field
  // if (selected_columns) {
  //   selected_columns.forEach((column, index) => {
  //     formData.append(`selected_columns`, column);
  //   });
  // }
  // if (facilities_Token) {
  //   formData.append("facilities_Token", facilities_Token);
  // }
  if (layout_file && layout_file.length > 0) {
    formData.append("format_file", layout_file[0]); // Assuming single file upload
  }
  if (commercial_file) {
    commercial_file.forEach((file, index) => {
      formData.append(`pdf_files`, file); // Adjusted field name
    });
  }

  // // Append each item in fileColumns with the appropriate filename
  // Object.keys(fileColumns).forEach((filename) => {
  //   fileColumns[filename].forEach((column, index) => {
  //     formData.append(`file_columns_${filename}`, column);
  //   });
  // });

  try {
    const response = await http.post("/facilities_process", formData, {
      headers: {
        "Content-Type": "multipart/form-data",
        "Access-Control-Allow-Origin": "*",
        Accept: "application/json",
      },
    });

    if (response.status !== 200) {
      throw new Error(`Failed to trigger /process. Status: ${response.status}`);
    }

    // Sanitize the JSON response to replace NaN with null
    // const sanitizeJSON = (jsonString: string) => {
    //   return jsonString.replace(/NaN/g, "null");
    // };

    // const sanitizedResponse = sanitizeJSON(response.data);
    // const parsedData = JSON.parse(sanitizedResponse);

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
      api_Message([error.message, "r#EF3934"]);
      setTimeout(() => {
        api_Message([]);
      }, 3000);
    }
    console.error("Error uploading files:", error.message);
    throw error; // Throw or handle error as needed
  }
};

export default FacilitiesMultiplePostFile;

//For excel file upload

// import http from "./comercialhttp";

// const CommercialPostFile = async (
//   layout_file: File[] | null,
//   selected_columns: string[],
//   commercial_file: File[] | null,
//   fileColumns: { [key: string]: string[] }
// ) => {
//   const formData = new FormData();
//   formData.append("choice", "with_format");

//   if (layout_file && layout_file.length > 0) {
//     formData.append("layout_file", layout_file[0]); // Assuming single file upload
//   }

//   // Append each item in selected_columns as a separate field
//   if (selected_columns) {
//     selected_columns.forEach((column, index) => {
//       formData.append(`selected_columns`, column);
//     });
//   }

//   if (commercial_file) {
//     commercial_file.forEach((file, index) => {
//       formData.append(`comparison_files`, file); // Adjusted field name
//     });
//   }

//   // Append each item in fileColumns with the appropriate filename
//   Object.keys(fileColumns).forEach((filename) => {
//     fileColumns[filename].forEach((column, index) => {
//       formData.append(`file_columns_${filename}`, column);
//     });
//   });

//   try {
//     const response = await http.post("/process", formData, {
//       headers: {
//         "Content-Type": "multipart/form-data",
//         "Access-Control-Allow-Origin": "*",
//         Accept: "application/json",
//       },
//       responseType: "blob", // Set response type to blob
//     });

//     if (response.status !== 200) {
//       throw new Error(`Failed to trigger /process. Status: ${response.status}`);
//     }

//     // Handle blob response
//     const blob = response.data;
//     const contentDisposition = response.headers["content-disposition"];
//     let filename = "downloaded_file"; // Default filename

//     if (contentDisposition) {
//       filename = contentDisposition.split("filename=")[1].replace(/"/g, "");
//     } else {
//       console.warn(
//         "Content-Disposition header is missing, using default filename"
//       );
//     }

//     const link = document.createElement("a");
//     link.href = window.URL.createObjectURL(blob);
//     link.download = filename;
//     document.body.appendChild(link); // Append to body
//     link.click();
//     document.body.removeChild(link); // Remove from body

//     return response.data; // Return data or handle as needed
//   } catch (error: any) {
//     console.error("Error uploading files:", error.message);
//     throw error; // Throw or handle error as needed
//   }
// };

// export default CommercialPostFile;