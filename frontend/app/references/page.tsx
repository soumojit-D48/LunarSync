import type { Metadata } from "next";
import ReferencesPage from "./ReferencesClient";

export const metadata: Metadata = {
  title: "References — LunarSync",
  description: "Seed the cloud reference set used for matching in cloud mode.",
};

export default function Page() {
  return <ReferencesPage />;
}
