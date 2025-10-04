// http.tsx
import axios from "axios";
import { EXCEL_API_BASE_URL } from "../config";

const http = axios.create({
  baseURL: `${EXCEL_API_BASE_URL}`,
});

export default http;
