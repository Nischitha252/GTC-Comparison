//config.tsx
import { Configuration, PopupRequest } from "@azure/msal-browser";

// Config object to be passed to Msal on creation
export const msalConfig: Configuration = {
  auth: {
    authority:
      "https://login.microsoftonline.com/372ee9e0-9ce0-4033-a64a-c07073a91ecd",
    clientId: "1aa3214e-6d0d-4c28-86cf-97279f198914",
    redirectUri: document.getElementById("root")?.baseURI || "",
    postLogoutRedirectUri: "/",
  },
  system: {
    allowNativeBroker: false, // Disables WAM Broker
  },
};

// export const EXCEL_API_BASE_URL =
//   "http://127.0.0.1:5000";
export const EXCEL_API_BASE_URL = "";

export const MICROSOFT_FORMS_URL = "https://forms.microsoft.com/e/dtUU86f242";

export const CONTACT_US = "mailto:venkatesha.vc@in.abb.com";

export const BLOB_STORAGE_URL =
  "https://imsindirectstorage.blob.core.windows.net/downloadedfiles";
